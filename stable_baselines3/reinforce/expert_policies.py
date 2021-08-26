import random

import torch as th


def continuous_mountain_car_expert_policy(time_step: int, add_noise: bool) -> th.Tensor:
    """
    This function is used to generate expert trajectories on Mountain Car environments
    so as to circumvent a deceptive gradient effect
    that makes standard policy gradient very inefficient on these environments.

    :param time_step: current episode step
    :param add_noise: whether to add some noise to the output
    ;return: an action depending on the current time step, eventually adding some noise
    """
    if add_noise:
        noise = random.random() / 20
    else:
        noise = 0.0
    if time_step < 50:
        return th.tensor([[-1.0 + noise]])
    else:
        return th.tensor([[1.0 + noise]])
