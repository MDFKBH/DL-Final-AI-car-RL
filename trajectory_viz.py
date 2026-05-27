"""Visualize policy trajectories overlaid on the map.

Usage:
    python trajectory_viz.py logs/ppo_full_map3_<ts>_s42/final_model.zip \
        --episodes 10 --out trajectory.png

Output: a PNG with the map as background and N colored trajectories drawn on top.
"""
import argparse
import json
import os
import sys

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model_path")
    p.add_argument("--map", default=None, help="Override map path")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=999)
    p.add_argument("--out", default="trajectory.png")
    p.add_argument("--linewidth", type=float, default=2.0)
    p.add_argument("--alpha", type=float, default=0.7)
    args = p.parse_args()

    # Headless before pygame import
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from stable_baselines3 import PPO
    from car_env import CarRacingEnv
    from configs import MAP_INFO

    # Find exp_config.json
    model_dir = os.path.dirname(os.path.abspath(args.model_path))
    cfg_path = None
    for d in [model_dir, os.path.dirname(model_dir)]:
        cand = os.path.join(d, "exp_config.json")
        if os.path.exists(cand):
            cfg_path = cand
            break
    if cfg_path is None:
        sys.exit("exp_config.json not found near model")
    with open(cfg_path) as f:
        exp_cfg = json.load(f)

    map_path = args.map or exp_cfg["map"]
    info = MAP_INFO.get(os.path.basename(map_path), {})

    env = CarRacingEnv(
        map_path=map_path,
        reward_config=exp_cfg["reward_config"],
        obs_components=exp_cfg["obs_components"],
        finish_line=info.get("finish_line"),
        forward_dir=info.get("forward_dir", (1, 0)),
        start_pos=info.get("start_pos", (830, 920)),
        start_angle=info.get("start_angle", 0.0),
    )
    model = PPO.load(args.model_path)

    # Collect trajectories
    trajectories = []
    crashed_flags = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        path = [info.get("start_pos", (830, 920))]
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, step_info = env.step(action)
            path.append(step_info["center"])
            if terminated or truncated:
                crashed_flags.append(step_info["crashed"])
                break
        trajectories.append(np.array(path))
    env.close()

    # Plot
    map_img = np.array(Image.open(map_path))
    h, w = map_img.shape[:2]

    fig, ax = plt.subplots(figsize=(12, 12 * h / w), dpi=100)
    ax.imshow(map_img)

    # Distinct colors for each episode
    cmap = plt.get_cmap("viridis")
    for i, traj in enumerate(trajectories):
        color = cmap(i / max(args.episodes - 1, 1))
        ax.plot(traj[:, 0], traj[:, 1],
                color=color, linewidth=args.linewidth, alpha=args.alpha,
                label=f"ep{i} ({'crash' if crashed_flags[i] else 'OK'})")
        # Crash marker
        if crashed_flags[i]:
            ax.scatter(traj[-1, 0], traj[-1, 1], color="red", marker="x", s=120, zorder=5)

    # Start point
    sx, sy = info.get("start_pos", (830, 920))
    ax.scatter(sx, sy, color="cyan", marker="o", s=200, edgecolors="black",
               linewidths=2, zorder=6, label="start")

    # Finish line
    fl = info.get("finish_line")
    if fl is not None:
        p1, p2 = fl
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="lime", linewidth=4,
                label="finish", zorder=4)

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
    ax.set_title(f"{os.path.basename(args.model_path)} on {os.path.basename(map_path)} "
                 f"({args.episodes} eps, {sum(crashed_flags)}/{args.episodes} crashed)",
                 fontsize=11)

    plt.tight_layout()
    plt.savefig(args.out, dpi=120, bbox_inches="tight", facecolor="white")
    print(f"Saved {args.out}")
    print(f"Crashed: {sum(crashed_flags)}/{args.episodes}")


if __name__ == "__main__":
    main()
