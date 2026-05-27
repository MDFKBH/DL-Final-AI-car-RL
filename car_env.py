"""Gymnasium env wrapping the self-driving car simulation. Modular obs + reward."""
import math
import os
from collections import deque

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

WIDTH, HEIGHT = 1920, 1080
CAR_SIZE_X, CAR_SIZE_Y = 60, 60
BORDER_COLOR = np.array([255, 255, 255], dtype=np.uint8)
MAX_RADAR_LEN = 300
MAX_SPEED = 40
RADAR_ANGLES = list(range(-90, 120, 45))  # 5 radars
ACTION_HISTORY_LEN = 4
NUM_ACTIONS = 4

VALID_REWARD_KEYS = {"progress", "crash", "speed", "smooth", "center"}
VALID_OBS_KEYS = {"radar", "speed", "angle", "action_history"}


class CarRacingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        map_path="assets/maps/map.png",
        car_path="assets/car.png",
        start_pos=(830, 920),
        start_angle=0.0,
        max_steps=1500,
        reward_config=None,
        obs_components=None,
        finish_line=None,        # ((x1,y1),(x2,y2)) or None
        forward_dir=(1, 0),
        render_mode=None,
    ):
        super().__init__()
        self.map_path = map_path
        self.start_pos = tuple(start_pos)
        self.start_angle = float(start_angle)
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.finish_line = finish_line
        self.forward_dir = forward_dir

        self.reward_config = {k: False for k in VALID_REWARD_KEYS}
        self.reward_config.update(reward_config or {"progress": True})
        bad = set(self.reward_config) - VALID_REWARD_KEYS
        if bad:
            raise ValueError(f"Unknown reward keys: {bad}")

        self.obs_components = list(obs_components or ["radar"])
        bad = set(self.obs_components) - VALID_OBS_KEYS
        if bad:
            raise ValueError(f"Unknown obs components: {bad}")

        if not pygame.get_init():
            pygame.init()
        self._surface = pygame.display.set_mode((WIDTH, HEIGHT))
        self.map_surface = pygame.image.load(map_path).convert()
        self.map_array = pygame.surfarray.array3d(self.map_surface)
        self.sprite = pygame.image.load(car_path).convert()
        self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X, CAR_SIZE_Y))

        # Build obs space dynamically
        obs_dim = 0
        if "radar" in self.obs_components:           obs_dim += len(RADAR_ANGLES)
        if "speed" in self.obs_components:           obs_dim += 1
        if "angle" in self.obs_components:           obs_dim += 2
        if "action_history" in self.obs_components:  obs_dim += ACTION_HISTORY_LEN * NUM_ACTIONS
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        self._font = None
        self._clock = None
        self._init_state()

    # ---------- helpers ----------
    def _init_state(self):
        self.position = list(self.start_pos)
        self.angle = self.start_angle
        self.speed = 20.0
        self.alive = True
        self.distance = 0.0
        self.time = 0
        self.center = [self.position[0] + CAR_SIZE_X / 2, self.position[1] + CAR_SIZE_Y / 2]
        self.radars = [MAX_RADAR_LEN] * len(RADAR_ANGLES)
        self.prev_action = -1
        self.action_hist = deque([0] * ACTION_HISTORY_LEN, maxlen=ACTION_HISTORY_LEN)
        self.lap_count = 0
        self.lap_start_step = 0
        self.last_lap_time = None
        self._prev_center = tuple(self.center)
        self._lap_grace_steps = 5   # ignore lap detection for first 5 steps after reset

    def _is_border(self, x, y):
        x, y = int(x), int(y)
        if x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT:
            return True
        return bool(np.array_equal(self.map_array[x, y], BORDER_COLOR))

    def _check_radar(self, degree):
        rad = math.radians(360 - (self.angle + degree))
        cos_r, sin_r = math.cos(rad), math.sin(rad)
        length = 0
        while length < MAX_RADAR_LEN:
            x = int(self.center[0] + cos_r * length)
            y = int(self.center[1] + sin_r * length)
            if self._is_border(x, y):
                break
            length += 1
        return length

    def _check_collision(self):
        half = 0.5 * CAR_SIZE_X
        for offset in (30, 150, 210, 330):
            rad = math.radians(360 - (self.angle + offset))
            cx = self.center[0] + math.cos(rad) * half
            cy = self.center[1] + math.sin(rad) * half
            if self._is_border(cx, cy):
                return True
        return False

    def _compute_radars(self):
        self.radars = [self._check_radar(d) for d in RADAR_ANGLES]

    @staticmethod
    def _segments_cross(a, b, c, d):
        def ccw(p, q, r):
            return (r[1] - p[1]) * (q[0] - p[0]) > (q[1] - p[1]) * (r[0] - p[0])
        return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)

    def _check_lap(self):
        if self.finish_line is None or self.time < self._lap_grace_steps:
            return False
        p1, p2 = self.finish_line
        if not self._segments_cross(self._prev_center, tuple(self.center), p1, p2):
            return False
        dx = self.center[0] - self._prev_center[0]
        dy = self.center[1] - self._prev_center[1]
        return (dx * self.forward_dir[0] + dy * self.forward_dir[1]) > 0

    # ---------- obs / reward ----------
    def _build_obs(self):
        parts = []
        if "radar" in self.obs_components:
            parts.append(np.array(self.radars, dtype=np.float32) / MAX_RADAR_LEN)
        if "speed" in self.obs_components:
            parts.append(np.array([self.speed / MAX_SPEED], dtype=np.float32))
        if "angle" in self.obs_components:
            rad = math.radians(self.angle)
            parts.append(np.array([math.sin(rad), math.cos(rad)], dtype=np.float32))
        if "action_history" in self.obs_components:
            hist = np.zeros((ACTION_HISTORY_LEN, NUM_ACTIONS), dtype=np.float32)
            for i, a in enumerate(self.action_hist):
                hist[i, a] = 1.0
            parts.append(hist.flatten())
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)

    def _compute_reward(self, crashed, action):
        r = 0.0
        comps = {}
        cfg = self.reward_config
        if cfg["progress"]:
            v = self.speed / 30.0
            comps["progress"] = v; r += v
        if cfg["crash"] and crashed:
            v = -1.0
            comps["crash"] = v; r += v
        if cfg["speed"]:
            v = 0.01 * (self.speed - 12) / max(MAX_SPEED - 12, 1)
            comps["speed"] = v; r += v
        if cfg["smooth"]:
            v = -0.05 if (self.prev_action != -1 and action != self.prev_action) else 0.0
            comps["smooth"] = v; r += v
        if cfg["center"]:
            left, right = self.radars[0], self.radars[-1]
            denom = max(left + right, 1)
            v = 0.1 * (1.0 - abs(left - right) / denom)
            comps["center"] = v; r += v
        return r, comps

    # ---------- gym API ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._init_state()
        # Randomize start angle for eval robustness
        self.angle = self.start_angle + self.np_random.uniform(-15, 15)
        self._compute_radars()
        return self._build_obs(), {}

    def step(self, action):
        action = int(action)
        if action == 0:   self.angle += 10
        elif action == 1: self.angle -= 10
        elif action == 2:
            if self.speed - 2 >= 12: self.speed -= 2
        elif action == 3:
            if self.speed + 2 <= MAX_SPEED: self.speed += 2

        self._prev_center = tuple(self.center)
        rad = math.radians(360 - self.angle)
        self.position[0] += math.cos(rad) * self.speed
        self.position[1] += math.sin(rad) * self.speed
        self.position[0] = max(20, min(self.position[0], WIDTH - 120))
        self.position[1] = max(20, min(self.position[1], HEIGHT - 120))
        self.center = [self.position[0] + CAR_SIZE_X / 2, self.position[1] + CAR_SIZE_Y / 2]
        self.distance += self.speed
        self.time += 1

        crashed = self._check_collision()
        self.alive = not crashed
        self._compute_radars()
        reward, comps = self._compute_reward(crashed, action)

        # Lap detection
        lap_completed = self._check_lap()
        if lap_completed:
            self.lap_count += 1
            self.last_lap_time = self.time - self.lap_start_step
            self.lap_start_step = self.time

        self.prev_action = action
        self.action_hist.append(action)

        terminated = crashed
        truncated = self.time >= self.max_steps
        info = {
            "distance": self.distance,
            "time": self.time,
            "speed": self.speed,
            "crashed": crashed,
            "lap_count": self.lap_count,
            "last_lap_time": self.last_lap_time,
            "lap_completed": lap_completed,
            "reward_components": comps,
            "center": tuple(self.center),
        }
        return self._build_obs(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return None
        if self._font is None:
            self._font = pygame.font.SysFont("Arial", 20)
        if self._clock is None:
            self._clock = pygame.time.Clock()
        self._surface.blit(self.map_surface, (0, 0))
        rot = pygame.transform.rotate(self.sprite, self.angle)
        rect = rot.get_rect(center=(int(self.center[0]), int(self.center[1])))
        self._surface.blit(rot, rect.topleft)
        for d, length in zip(RADAR_ANGLES, self.radars):
            rad = math.radians(360 - (self.angle + d))
            end = (int(self.center[0] + math.cos(rad) * length),
                   int(self.center[1] + math.sin(rad) * length))
            pygame.draw.line(self._surface, (0, 255, 0), self.center, end, 1)
            pygame.draw.circle(self._surface, (0, 255, 0), end, 5)
        if self.finish_line is not None:
            pygame.draw.line(self._surface, (255, 0, 0), self.finish_line[0], self.finish_line[1], 3)
        if self.render_mode == "human":
            pygame.display.flip()
            self._clock.tick(60)
            return None
        arr = pygame.surfarray.array3d(self._surface)
        return np.transpose(arr, (1, 0, 2))

    def close(self):
        pygame.quit()