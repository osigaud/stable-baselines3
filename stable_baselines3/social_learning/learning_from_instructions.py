import os

import gym

from chrono import Chrono  # To measure time in human readable format, use stop() to display time since chrono creation

from visu.visu_critics import plot_2d_critic  # Function to plot critics
from visu.visu_policies import plot_2d_policy  # Function to plot policies
from stable_baselines3 import REINFORCE
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from visu.visu_trajectories import plot_trajectory
from tutor_instruction_wrapper import TutorInstructionWrapper

chrono = Chrono()
env_name = "MountainCarContinuous-v0"
env = gym.make(env_name)
env_vec = make_vec_env(env_name, n_envs=10, seed=0)

log_dir = "data/save/"
os.makedirs(log_dir, exist_ok=True)

policy_kwargs = dict(net_arch=dict(pi=[100, 100], vf=[100, 100]), optimizer_kwargs=dict(eps=1e-5))
gradient_name="discount"

file_name = gradient_name + "_" + env_name
log_file_name = log_dir + file_name

eval_callback = EvalCallback(
            env_vec,
            best_model_save_path=log_dir + "bests/",
            log_path=log_dir,
            eval_freq=500,
            n_eval_episodes=50,
            deterministic=True,
            render=False,
        )


env = TutorInstructionWrapper(env)
model = REINFORCE(
            "MlpPolicy",
            env,
            gradient_name=gradient_name,
            beta=1,
            gamma=0.99,
            learning_rate=0.0001,
            n_steps=5,
            seed=1,
            verbose=1,
            policy_kwargs=policy_kwargs,
            tensorboard_log=log_file_name,
            critic_estim_method="td",
            n_critic_epochs=25,
        )
model.learn(
            total_timesteps=20000,
            nb_rollouts=25,
            callback=eval_callback,
            log_interval=20,
        )

print("nb total instructions:", env.nb_total_instructions)
chrono.stop()

rollout_data = model.rollout_buffer.get_samples()
plot_trajectory(rollout_data, env, 1, plot=True)