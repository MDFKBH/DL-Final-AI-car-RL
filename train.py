"""PPO training with locked hyperparams. All variables come from configs.py."""
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"  # MUST be set before any pygame import

import argparse
import json
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from car_env import CarRacingEnv
from configs import EXPERIMENTS, PPO_HYPERPARAMS, MAP_INFO


def make_env(map_path, exp_cfg, seed):
    map_basename = os.path.basename(map_path)
    info = MAP_INFO.get(map_basename, {})

    def _init():
        env = CarRacingEnv(
            map_path=map_path,
            reward_config=exp_cfg["reward"],
            obs_components=exp_cfg["obs"],
            finish_line=info.get("finish_line"),
            forward_dir=info.get("forward_dir", (1, 0)),
            start_pos=info.get("start_pos", (830, 920)),
            start_angle=info.get("start_angle", 0.0),
        )
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", default="baseline", choices=list(EXPERIMENTS.keys()))
    p.add_argument("--map", default="assets/maps/map.png")
    p.add_argument("--timesteps", type=int, default=500_000)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--vec", default="subproc", choices=["dummy", "subproc"])
    args = p.parse_args()

    exp_cfg = EXPERIMENTS[args.exp]
    map_name = os.path.splitext(os.path.basename(args.map))[0]
    run_name = f"ppo_{args.exp}_{map_name}_{datetime.now():%Y%m%d_%H%M%S}_s{args.seed}"
    log_dir = f"./logs/{run_name}"
    os.makedirs(log_dir, exist_ok=True)

    VecCls = SubprocVecEnv if (args.vec == "subproc" and args.n_envs > 1) else DummyVecEnv
    env = VecCls([make_env(args.map, exp_cfg, args.seed + i) for i in range(args.n_envs)])

    model = PPO("MlpPolicy", env, verbose=1,
                tensorboard_log="./logs/tb/", seed=args.seed, **PPO_HYPERPARAMS)

    ckpt_cb = CheckpointCallback(
        save_freq=max(50_000 // args.n_envs, 1),
        save_path=f"{log_dir}/checkpoints",
        name_prefix="ppo",
    )
    model.learn(total_timesteps=args.timesteps, tb_log_name=run_name, callback=[ckpt_cb])
    model.save(f"{log_dir}/final_model")

    with open(f"{log_dir}/exp_config.json", "w") as f:
        json.dump({
            "experiment": args.exp, "map": args.map,
            "timesteps": args.timesteps, "n_envs": args.n_envs, "seed": args.seed,
            "reward_config": exp_cfg["reward"], "obs_components": exp_cfg["obs"],
            "hyperparams": {k: v for k, v in PPO_HYPERPARAMS.items() if k != "policy_kwargs"},
            "policy_kwargs": PPO_HYPERPARAMS["policy_kwargs"],
        }, f, indent=2)
    print(f"Saved to {log_dir}/")


if __name__ == "__main__":
    main()