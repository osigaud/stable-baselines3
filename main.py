import os
import random

import gym
import my_gym  # Necessary to see CartPoleContinuous, though PyCharm does not understand this
import numpy as np
import torch
from arguments import get_args
from chrono import Chrono
from visu.visu_critics import plot_cartpole_critic, plot_pendulum_critic
from visu.visu_policies import plot_2d_policy, plot_cartpole_policy, plot_pendulum_policy

from stable_baselines3 import REINFORCE
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

# from stable_baselines3.reinforce.custom_monitor import CustomMonitor
# from stable_baselines3.reinforce.loss_callback import LossCallback

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


def plot_policies(model, env, env_name, gradient_name, final_string="post"):
    actname = env_name + "_actor_" + gradient_name + "_" + final_string + ".pdf"
    critname = env_name + "_critic_" + gradient_name + "_" + final_string + ".pdf"
    if env_name == "Pendulum-v0":
        plot_pendulum_policy(model.policy, env, deterministic=True, figname=actname, plot=False)
        plot_pendulum_critic(model.policy, env, deterministic=True, figname=critname, plot=False)
    elif env_name == "CartPole-v1" or env_name == "CartPoleContinuous-v0":
        plot_cartpole_policy(model.policy, env, deterministic=True, figname=actname, plot=False)
        plot_cartpole_critic(model.policy, env, deterministic=True, figname=critname, plot=False)
    else:
        plot_2d_policy(model.policy, env, deterministic=True)

def test_reinforce() -> None:
    plot_policies = False
    args = get_args()
    chrono = Chrono()
    # Create log dir
    log_dir = "data/save/"
    os.makedirs(log_dir, exist_ok=True)
    # args.env_name = "Pendulum-v0"
    args.env_name = "CartPole-v1"
    args.gradients = ["discount","normalized discount"]
    # args.gradients = ["sum","discount","normalize","normalized discount","n step","baseline","gae"]
    args.nb_rollouts = 2
    # Create and wrap the environment
    env = gym.make(args.env_name)
    # eval_env = gym.make(args.env_name)
    # env_wrapped = Monitor(eval_env, log_dir)
    env_vec = make_vec_env(args.env_name, n_envs=10, seed=0, vec_env_cls=DummyVecEnv)
    grads = args.gradients
    for i in range(len(grads)):
        file_name = grads[i] + "_" + args.env_name
        log_file_name = log_dir + file_name
        print(grads[i])
        # lcb = LossCallback(log_dir, file_name)
        eval_callback = EvalCallback(
            env_vec,
            best_model_save_path=log_dir + "bests/",
            log_path=log_dir,
            eval_freq=500,
            n_eval_episodes=50,
            deterministic=True,
            render=False,
        )

        model = REINFORCE(
            "MlpPolicy",
            env,
            grads[i],
            beta=args.beta,
            nb_rollouts=args.nb_rollouts,
            seed=1,
            verbose=1,
            tensorboard_log=log_file_name,
        )

        if plot_policies:
            plot_policies(model, env, args.env_name, grads[i], final_string="pre")

        for rep in range(args.nb_repet):
            model.learn(int(3000), reset_num_timesteps=rep == 0, callback=eval_callback, log_interval=args.log_interval)

        if plot_policies:
            plot_policies(model, env, args.env_name, grads[i], final_string="post")

    chrono.stop()
    # plot_results(args)


def test2():
    model = REINFORCE(
        "MlpPolicy",
        "CartPoleContinuous-v0",
        gradient_name="gae",
        seed=1,
        verbose=1,
    )
    model.learn(int(1e5))


if __name__ == "__main__":
    # test2()
    test_reinforce()
