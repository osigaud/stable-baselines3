import os
import random
from typing import Any, Dict, List, Optional, TextIO, Tuple, Type, Union

import gym
import my_gym  # Necessary to see CartPoleContinuous, though PyCharm does not understand this
import numpy as np
import torch
from arguments import get_args
from chrono import Chrono
from visu.visu_critics import plot_pendulum_critic, plot_cartpole_critic, plot_2d_critic
from visu.visu_policies import plot_pendulum_policy, plot_cartpole_policy, plot_2d_policy
from visu.visu_results import plot_results

from stable_baselines3 import A2C, REINFORCE, TD3
from stable_baselines3.reinforce.custom_monitor import CustomMonitor
from stable_baselines3.reinforce.loss_callback import LossCallback

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


def create_data_folders() -> None:
    """
    Create folders where to save output files if they are not already there
    :return: nothing
    """
    if not os.path.exists("data/save"):
        os.mkdir("./data")
        os.mkdir("./data/save")
    if not os.path.exists("data/critics"):
        os.mkdir("./data/critics")
    if not os.path.exists("data/policies/"):
        os.mkdir("data/policies/")
    if not os.path.exists("data/results/"):
        os.mkdir("data/results/")


def set_files(study_name, env_name) -> Tuple[TextIO, TextIO]:
    """
    Create files to save the policy loss and the critic loss
    :param study_name: the name of the study
    :param env_name: the name of the environment
    :return:
    """
    policy_loss_name = "data/save/policy_loss_" + study_name + "_" + env_name + ".txt"
    policy_loss_file = open(policy_loss_name, "w")
    critic_loss_name = "data/save/critic_loss_" + study_name + "_" + env_name + ".txt"
    critic_loss_file = open(critic_loss_name, "w")
    return policy_loss_file, critic_loss_file


def test_reinforce() -> None:
    args = get_args()
    chrono = Chrono()
    # Create log dir
    log_dir = "data/save/"
    os.makedirs(log_dir, exist_ok=True)
    # args.env_name = "Pendulum-v0"
    args.env_name = "CartPole-v1"
    env_name = args.env_name

    # Create and wrap the environment
    env_init = gym.make(env_name)
    grads = args.gradients
    # grads = ["gae"]
    nb_repet = 5
    args.nb_rollouts = 2
    for i in range(len(grads)):
        file_name = grads[i] + "_" + env_name
        print(grads[i])
        lcb = LossCallback(log_dir, file_name)
        env = CustomMonitor(env_init, log_dir, file_name)
        model = REINFORCE("MlpPolicy", env, grads[i], args.beta, args.nb_rollouts, seed=1, verbose=1)

        actname = args.env_name + "_actor_" + grads[i] + "_pre.pdf"
        if args.env_name == "Pendulum-v0":
            plot_pendulum_policy(model.policy, env_init, deterministic=True, figname=actname)
        elif args.env_name == "CartPole-v1" or args.env_name == "CartPoleContinuous-v0":
            plot_cartpole_policy(model.policy, env_init, deterministic=True, figname=actname)
        else:
            plot_2d_policy(model.policy, env_init, deterministic=True)
        for rep in range(nb_repet):
            env.start_again()
            model.reset_episodes()
            model.learn(int(6000), reset_num_timesteps=rep == 0, callback=lcb)

        actname = args.env_name + "_actor_" + grads[i] + "_post.pdf"
        critname = args.env_name + "_critic_" + grads[i] + "_post.pdf"
        if args.env_name == "Pendulum-v0":
            plot_pendulum_policy(model.policy, env_init, deterministic=True, figname=actname)
            plot_pendulum_critic(model.policy, env_init, deterministic=True, figname=critname)
        elif args.env_name == "CartPole-v1" or args.env_name == "CartPoleContinuous-v0":
            plot_cartpole_policy(model.policy, env_init, deterministic=True, figname=actname)
            plot_cartpole_critic(model.policy, env_init, deterministic=True, figname=critname)
        else:
            plot_2d_policy(model.policy, env_init, deterministic=True)
    chrono.stop()
    plot_results(args)


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
