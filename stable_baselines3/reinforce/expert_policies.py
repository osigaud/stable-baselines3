import random
import torch as th

def continuous_mountain_car_expert_policy(time_step, var):
    """
    This function is used to generate expert trajectories on Mountain Car environments
    so as to circumvent a deceptive gradient effect
    that makes standard policy gradient very inefficient on these environments.
    Return an action depending on the current time step, eventually adding some noise
    """
    if var:
        variation = random.random() / 20
    else:
        variation = 0
    if time_step < 50:
        return th.tensor([[-1 + variation]])
    else:
        return th.tensor([[1 + variation]])

