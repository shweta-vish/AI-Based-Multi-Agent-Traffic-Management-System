# train_agent.py  (FIXED — works with your TrafficEnv observation shape of 6)
# ─────────────────────────────────────────────────────────────────────────────
# Run this FIRST before benchmark_compare.py
# Output: traffic_dqn_v2.zip  +  vec_normalize.pkl  +  ./best_model/
# ─────────────────────────────────────────────────────────────────────────────

import os
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from traffic_env import TrafficEnv

# ── 1. Sanity-check the env ───────────────────────────────────────────────────
print("Checking environment ...")
check_env(TrafficEnv(), warn=True)
print("Environment OK.\n")

# ── 2. Vectorised + normalised envs ──────────────────────────────────────────
def make_env():
    return TrafficEnv()

train_env = DummyVecEnv([make_env])
train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=50.0)

eval_raw = DummyVecEnv([make_env])
eval_env = VecNormalize(eval_raw, norm_obs=True, norm_reward=False, clip_obs=50.0)

# ── 3. Model ──────────────────────────────────────────────────────────────────
model = DQN(
    policy="MlpPolicy",
    env=train_env,
    learning_rate=5e-4,
    buffer_size=100_000,
    learning_starts=2_000,
    batch_size=128,
    gamma=0.95,
    train_freq=1,
    target_update_interval=200,
    exploration_fraction=0.4,
    exploration_final_eps=0.05,
    policy_kwargs=dict(net_arch=[128, 128, 64]),
    verbose=1,
    tensorboard_log="./tb_logs/",
)

# ── 4. Callbacks: save best model + periodic checkpoint ──────────────────────
os.makedirs('best_model', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path='./best_model/',
    log_path='./logs/',
    eval_freq=10_000,          # evaluate every 10k steps
    n_eval_episodes=5,
    deterministic=True,
    render=False,
    verbose=1,
)
checkpoint_callback = CheckpointCallback(
    save_freq=50_000,
    save_path='./checkpoints/',
    name_prefix='dqn_traffic',
)

# ── 5. Train ──────────────────────────────────────────────────────────────────
TIMESTEPS = 100_000
print(f"Training DQN for {TIMESTEPS:,} steps (~10-20 mins) ...")
model.learn(total_timesteps=TIMESTEPS,
            callback=[eval_callback, checkpoint_callback])

model.save("traffic_dqn_v2")
train_env.save("vec_normalize.pkl")
print("\nModel saved → traffic_dqn_v2.zip")
print("Normalisation stats → vec_normalize.pkl")

# ── 6. Final evaluation ───────────────────────────────────────────────────────
eval_env2 = DummyVecEnv([make_env])
eval_env2 = VecNormalize.load("vec_normalize.pkl", eval_env2)
eval_env2.training   = False
eval_env2.norm_reward = False

mean_reward, std_reward = evaluate_policy(model, eval_env2, n_eval_episodes=20)
print(f"\nFinal eval — Mean reward: {mean_reward:.1f} ± {std_reward:.1f}")
print("Training complete! Now run:  python benchmark_compare.py --episodes 10")
