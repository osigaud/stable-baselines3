import gym
import matplotlib.pyplot as plt
import numpy as np
import torch as th

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

tensorboard_log = "data/tb/"

env = gym.make("MountainCar-v0")
dqn_model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    train_freq=16,
    gradient_steps=8,
    gamma=0.99,
    exploration_fraction=0.2,
    exploration_final_eps=0.07,
    target_update_interval=600,
    learning_starts=1000,
    buffer_size=10000,
    batch_size=128,
    learning_rate=4e-3,
    policy_kwargs=dict(net_arch=[256, 256]),
    tensorboard_log=tensorboard_log,
    seed=2,
)

dqn_model.learn(int(1.2e4), log_interval=10)

mean_reward, std_reward = evaluate_policy(dqn_model, dqn_model.get_env(), deterministic=True, n_eval_episodes=20)

print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
