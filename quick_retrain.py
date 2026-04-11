# quick_retrain.py
# Trains a fresh DQN on the FIXED headless_sim in ~5-10 minutes
# Run this, then run benchmark_compare.py

import os
import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from traffic_env import TrafficEnv

# ── delete old broken models ──────────────────────────────────────────────────
for f in ['traffic_dqn_v2.zip', 'vec_normalize.pkl']:
    if os.path.exists(f):
        os.remove(f)
        print(f"Deleted old: {f}")
for f in ['best_model/best_model.zip', 'best_model/best_model']:
    if os.path.exists(f):
        os.remove(f)
        print(f"Deleted old: {f}")

os.makedirs('best_model', exist_ok=True)

# ── env ───────────────────────────────────────────────────────────────────────
def make_env():
    return TrafficEnv()

train_env = DummyVecEnv([make_env])
train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=300.0)

eval_env = DummyVecEnv([make_env])
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=300.0)

# ── model ─────────────────────────────────────────────────────────────────────
model = DQN(
    "MlpPolicy",
    train_env,
    learning_rate=1e-3,
    buffer_size=50_000,
    learning_starts=1_000,
    batch_size=64,
    gamma=0.95,
    train_freq=1,
    target_update_interval=100,
    exploration_fraction=0.35,
    exploration_final_eps=0.05,
    policy_kwargs=dict(net_arch=[128, 128, 64]),
    verbose=1,
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path='./best_model/',
    eval_freq=5_000,
    n_eval_episodes=3,
    deterministic=True,
    verbose=1,
)

# ── train ─────────────────────────────────────────────────────────────────────
print("\nTraining on FIXED sim — ~5-10 minutes ...")
print("Watch for 'New best model' lines — that means it's learning.\n")

model.learn(total_timesteps=10_000, callback=eval_callback)
model.save("traffic_dqn_v2")
train_env.save("vec_normalize.pkl")

print("\nDone! Now run:  python benchmark_compare.py --episodes 10")
