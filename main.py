
import os
import random
import numpy as np
import torch
from typing import Any, Dict, List, Optional, Tuple, Type, Union, TextIO
from chrono import Chrono
from stable_baselines3 import REINFORCE

from arguments import get_args
from visu.visu_results import plot_results

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
    if not os.path.exists('data/policies/'):
        os.mkdir('data/policies/')
    if not os.path.exists('data/results/'):
        os.mkdir('data/results/')


def set_files(study_name, env_name) -> Tuple[TextIO,TextIO]:
    """
    Create files to save the policy loss and the critic loss
    :param study_name: the name of the study
    :param env_name: the name of the environment
    :return:
    """
    policy_loss_name = "data/save/policy_loss_" + study_name + '_' + env_name + ".txt"
    policy_loss_file = open(policy_loss_name, "w")
    critic_loss_name = "data/save/critic_loss_" + study_name + '_' + env_name + ".txt"
    critic_loss_file = open(critic_loss_name, "w")
    return policy_loss_file, critic_loss_file


if __name__ == '__main__':
    args = get_args()
    print(args)
    create_data_folders()
    chrono = Chrono()
    grad = args.gradients[0]
    model = REINFORCE('MlpPolicy', 'CartPole-v1', args.gradients[0], args.beta, args.nb_rollouts, args.max_episode_steps).learn(100)
    chrono.stop()
    plot_results(args)

