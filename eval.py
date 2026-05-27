"""Evaluate a trained model: lap time, crash rate, mean jerk, distance."""
import argparse
import json
import os

import numpy as np

from configs import MAP_INFO

def main():
    p = argparse.ArgumentParser()
    p.add_argument("model_path", help="Path to .zip model file")
    p.add_argument("--map", default=None)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--render", action="store_true")
    p.add_argument("--seed", type=int, default=999)
    p.add_argument("--deterministic", action="store_true", default=True)
    args = p.parse_args()

    # Headless unless rendering — set BEFORE importing pygame
    if not args.render:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    from stable_baselines3 import PPO
    from car_env import CarRacingEnv

    # Find exp_config.json (sibling or parent of model file)
    model_dir = os.path.dirname(os.path.abspath(args.model_path))
    cfg_path = None
    for d in [model_dir, os.path.dirname(model_dir)]:
        cand = os.path.join(d, "exp_config.json")
        if os.path.exists(cand):
            cfg_path = cand
            break
    if cfg_path is None:
        raise FileNotFoundError("exp_config.json not found near model")
    with open(cfg_path) as f:
        exp_cfg = json.load(f)

    map_path = args.map or exp_cfg["map"]
    map_basename = os.path.basename(map_path)
    info = MAP_INFO.get(map_basename, {})

    env = CarRacingEnv(
        map_path=map_path,
        reward_config=exp_cfg["reward_config"],
        obs_components=exp_cfg["obs_components"],
        finish_line=info.get("finish_line"),
        forward_dir=info.get("forward_dir", (1, 0)),
        start_pos=info.get("start_pos", (830, 920)),
        start_angle=info.get("start_angle", 0.0),
        render_mode="human" if args.render else None,
    )
    model = PPO.load(args.model_path)

    results = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        speeds, lap_times, positions = [], [], []
        prev_action = None
        action_changes = 0
        ep_len = 0
        crashed = False
        info = {}
        while True:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, _, terminated, truncated, info = env.step(action)
            speeds.append(info["speed"])
            positions.append(info["center"])
            if prev_action is not None and int(action) != prev_action:
                action_changes += 1
            prev_action = int(action)
            ep_len += 1
            if info.get("lap_completed") and info.get("last_lap_time"):
                lap_times.append(info["last_lap_time"])
            if args.render:
                env.render()
            if terminated or truncated:
                crashed = info["crashed"]
                break

        speeds = np.array(speeds)
        positions = np.array(positions)
        if len(positions) > 3:
            jerk_xy = np.diff(positions, n=3, axis=0)
            mean_jerk = float(np.mean(np.linalg.norm(jerk_xy, axis=1)))
        else:
            mean_jerk = 0.0


        results.append({
            "episode": ep,
            "length": ep_len,
            "crashed": bool(crashed),
            "distance": float(info["distance"]),
            "lap_count": int(info["lap_count"]),
            "lap_times": lap_times,
            "mean_speed": float(speeds.mean()),
            "action_switch_rate": action_changes / max(ep_len - 1, 1),
            "mean_jerk": mean_jerk,
        })
    env.close()

    # Aggregate
    n = len(results)
    crash_rate = sum(r["crashed"] for r in results) / n
    lengths = np.array([r["length"] for r in results])
    dists = np.array([r["distance"] for r in results])
    jerks = np.array([r["mean_jerk"] for r in results])
    asrs = np.array([r["action_switch_rate"] for r in results])
    all_lap_times = [t for r in results for t in r["lap_times"]]
    lap_summary = (f"{np.mean(all_lap_times):.1f} ± {np.std(all_lap_times):.1f} steps"
                   f" (n={len(all_lap_times)})") if all_lap_times else "N/A (no finish line set)"

    print(f"\n=== Eval over {n} episodes ===")
    print(f"  Crash rate:          {crash_rate:.1%}")
    print(f"  Lap time:            {lap_summary}")
    print(f"  Episode length:      {lengths.mean():.1f} ± {lengths.std():.1f}")
    print(f"  Distance/ep:         {dists.mean():.1f} ± {dists.std():.1f}")
    print(f"  Mean jerk:           {jerks.mean():.3f} ± {jerks.std():.3f}")
    print(f"  Action switch rate:  {asrs.mean():.2%} ± {asrs.std():.2%}")

    out = os.path.join(model_dir, "eval_results.json")
    with open(out, "w") as f:
        json.dump({
            "n_episodes": n, "seed": args.seed,
            "aggregate": {
                "crash_rate": crash_rate,
                "mean_episode_length": float(lengths.mean()),
                "std_episode_length": float(lengths.std()),
                "mean_distance": float(dists.mean()),
                "std_distance": float(dists.std()),
                "mean_jerk": float(jerks.mean()),
                "std_jerk": float(jerks.std()),
                "mean_action_switch_rate": float(asrs.mean()),
                "std_action_switch_rate": float(asrs.std()),
                "lap_times_mean": float(np.mean(all_lap_times)) if all_lap_times else None,
                "lap_times_std": float(np.std(all_lap_times)) if all_lap_times else None,
                "lap_completions_total": len(all_lap_times),
            },
            "per_episode": results,
        }, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()