"""All experiment variables live here. Edit ONE place per experiment."""
# === Map metadata (finish line, forward direction, start pose) ===
MAP_INFO = {
    "map.png": {
        "finish_line": ((831, 878), (830, 991)),
        "forward_dir": (1, 0),
        "start_pos": (830, 920),
        "start_angle": 0.0,
    },
    "map3.png": {
        "finish_line": ((830, 890), (832, 979)),
        "forward_dir": (1, 0),
        "start_pos": (830, 920),
        "start_angle": 0.0,
    },
}

# === LOCKED PPO hyperparams — DO NOT CHANGE between experiments ===
# These are SB3 defaults, well-tested for MlpPolicy on small discrete tasks.
PPO_HYPERPARAMS = dict(
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=dict(net_arch=[64, 64]),
)

# === Reward component toggles ===
# Each component has a fixed formula in car_env._compute_reward.
# Toggle on/off here; do NOT change the formula or weight between runs.
REWARD_BASELINE = {
    "progress": True,   # speed / 30 each step (mimics NEAT distance fitness)
    "crash":    False,  # -1 on crash
    "speed":    False,  # 0.01 * (speed - 12) / (max_speed - 12)
    "smooth":   False,  # -0.05 when action changes
    "center":   False,  # 0.1 * radar symmetry (|L - R| / (L + R))
}
REWARD_SHAPED = {"progress": True, "crash": True, "speed": True, "smooth": True, "center": True}

# === Observation component sets ===
OBS_BASELINE = ["radar"]                                       # 5 dim
OBS_FULL     = ["radar", "speed", "angle", "action_history"]   # 5 + 1 + 2 + 16 = 24 dim

# === Predefined experiments ===
EXPERIMENTS = {
    "baseline":    dict(reward=REWARD_BASELINE, obs=OBS_BASELINE),
    "reward_only": dict(reward=REWARD_SHAPED,   obs=OBS_BASELINE),
    "obs_only":    dict(reward=REWARD_BASELINE, obs=OBS_FULL),
    "full":        dict(reward=REWARD_SHAPED,   obs=OBS_FULL),
}