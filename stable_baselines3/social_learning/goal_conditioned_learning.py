import os

import gym
import numpy as np

from chrono import Chrono  # To measure time in human readable format, use stop() to display time since chrono creation

from visu.visu_critics import plot_2d_critic  # Function to plot critics
from visu.visu_policies import plot_2d_policy  # Function to plot policies
from custom_goal_env import CustomGoalEnv
from stable_baselines3 import TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from visu.visu_trajectories import plot_trajectory
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.env_checker import check_env

from stable_baselines3.common.monitor import Monitor
from gym.wrappers import TimeLimit

log_dir = "data/save/"
os.makedirs(log_dir, exist_ok=True)

policy_kwargs = dict(net_arch=dict(pi=[100, 100], vf=[100, 100]), optimizer_kwargs=dict(eps=1e-5))

env_name = "MountainCarContinuous-v0"
env = TimeLimit(CustomGoalEnv(env_name, True), 1000)
# env = CustomGoalEnv(env_name, True)
env_eval = Monitor(CustomGoalEnv(env_name, False))
check_env(env)
check_env(env_eval)

# env_vec = make_vec_env(env_name, n_envs=10, seed=0)

file_name = env_name
log_file_name = log_dir + file_name

eval_callback = EvalCallback(
            env_eval,
            best_model_save_path=log_dir + "bests/",
            log_path=log_dir,
            eval_freq=500,
            n_eval_episodes=5,
            deterministic=True,
            render=False,
        )

# Available strategies (cf paper): future, final, episode
goal_selection_strategy = 'future' # equivalent to GoalSelectionStrategy.FUTURE

# If True the HER transitions will get sampled online
online_sampling = True

model = TD3(
            "MultiInputPolicy",
            env,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=dict(
                n_sampled_goal=4,
                goal_selection_strategy=goal_selection_strategy,
                online_sampling=online_sampling,
                max_episode_length=1000,
            ),
            gamma=0.99,
            learning_rate=0.0001,
            seed=1,
            verbose=1,
            policy_kwargs=policy_kwargs,
            tensorboard_log=log_file_name,
        )
print("starting to learn")
model.learn(
            total_timesteps=100000,
            callback=eval_callback,
            log_interval=100,
        )

# rollout_data = model.replay_buffer._get_samples(np.array([0,model.replay_buffer.buffer_size]), env)
# plot_trajectory(rollout_data, env, 1, plot=True)