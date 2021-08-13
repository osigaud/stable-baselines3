import os
import random
import numpy as np
import torch

import gym
import my_gym  # Necessary to see CartPoleContinuous, though PyCharm does not understand this

from arguments import get_args
from chrono import Chrono
from visu.visu_critics import plot_cartpole_critic, plot_pendulum_critic
from visu.visu_policies import plot_2d_policy, plot_cartpole_policy, plot_pendulum_policy

from stable_baselines3 import REINFORCE
from stable_baselines3.reinforce.loss_callback import LossCallback
from stable_baselines3.common.callbacks import EvalCallback

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

def test_reinforce() -> None:
    args = get_args()
    chrono = Chrono()
    # Create log dir
    log_dir = "data/save/"
    os.makedirs(log_dir, exist_ok=True)
    # args.env_name = "Pendulum-v0"
    args.env_name = "CartPole-v1"
    args.gradients = ["baseline"]
    args.nb_rollouts = 2
    # Create and wrap the environment
    env = gym.make(args.env_name)
    grads = args.gradients
    for i in range(len(grads)):
        file_name = grads[i] + "_" + args.env_name
        log_file_name = log_dir + file_name
        print(grads[i])
        lcb = LossCallback(log_dir, file_name)
        eval_env = gym.make(args.env_name)
        eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir+'bests/',
                                     log_path=log_dir, eval_freq=500,
                                     deterministic=True, render=False)

        model = REINFORCE("MlpPolicy", env, grads[i], beta=args.beta, nb_rollouts=args.nb_rollouts, seed=1, verbose=1,
                tensorboard_log=log_file_name)

        actname = args.env_name + "_actor_" + grads[i] + "_pre.pdf"
        if args.env_name == "Pendulum-v0":
            plot_pendulum_policy(model.policy, env, deterministic=True, figname=actname, plot=False)
        elif args.env_name == "CartPole-v1" or args.env_name == "CartPoleContinuous-v0":
            plot_cartpole_policy(model.policy, env, deterministic=True, figname=actname, plot=False)
        else:
            plot_2d_policy(model.policy, env, deterministic=True)
        for rep in range(args.nb_repet):
            model.learn(int(3000), reset_num_timesteps=rep == 0, callback=eval_callback, log_interval=args.log_interval)

        actname = args.env_name + "_actor_" + grads[i] + "_post.pdf"
        critname = args.env_name + "_critic_" + grads[i] + "_post.pdf"
        if args.env_name == "Pendulum-v0":
            plot_pendulum_policy(model.policy, env, deterministic=True, figname=actname, plot=False)
            plot_pendulum_critic(model.policy, env, deterministic=True, figname=critname, plot=False)
        elif args.env_name == "CartPole-v1" or args.env_name == "CartPoleContinuous-v0":
            plot_cartpole_policy(model.policy, env, deterministic=True, figname=actname, plot=False)
            plot_cartpole_critic(model.policy, env, deterministic=True, figname=critname, plot=False)
        else:
            plot_2d_policy(model.policy, env, deterministic=True)
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
