import numpy as np
import torch as th

from gym import spaces
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from stable_baselines3.reinforce.episodic_buffer import EpisodicBuffer


class IncompleteBuffer(EpisodicBuffer):

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
        n_steps: int = 5,
        beta: float = 1.0,
        nb_rollouts: int = 1,
        max_episode_steps: int = 1,
        verbose=False,
    ):
        super(IncompleteBuffer, self).__init__(
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            gae_lambda=gae_lambda,
            gamma=gamma,
            n_envs=n_envs,
            n_steps=n_steps,
            beta=beta,
            nb_rollouts=nb_rollouts,
            max_episode_steps=max_episode_steps,
            verbose=verbose,
            )

    def get_discounted_sum_rewards(self) -> None:
        """
        Apply a discounted sum of rewards to all samples of all episodes
        """
        for ep in range(self.nb_rollouts):
            sum_discounted_rewards = 0
            for i in reversed(range(self.episode_lengths[ep])):
                sum_discounted_rewards = self.rewards[ep, i] + self.gamma * sum_discounted_rewards
                self.policy_returns[ep, i] = sum_discounted_rewards

    def get_sum_rewards(self) -> None:
        """
        Apply a sum of rewards to all samples of all episodes
        """
        # NOTE(antonin): to be completely correct, we should only
        # fill policy_returns until the episode_lengths
        self.policy_returns[:, :] = self.rewards.sum(axis=1, keepdims=True)

    def get_normalized_rewards(self) -> None:
        """
        Normalize rewards of all samples of all episodes
        """
        self.policy_returns = self.rewards.copy()
        self.policy_returns = (self.policy_returns - self.policy_returns.mean()) / (self.policy_returns.std() + 1e-8)

    def get_normalized_sum(self) -> None:
        """
        Normalize rewards of all samples of all episodes
        """
        self.get_sum_rewards()
        self.policy_returns = (self.policy_returns - self.policy_returns.mean()) / (self.policy_returns.std() + 1e-8)

    def get_normalized_discounted_rewards(self) -> None:
        """
        Apply a normalized and discounted sum of rewards to all samples of the episode
        """
        self.get_discounted_sum_rewards()
        self.policy_returns = (self.policy_returns - self.policy_returns.mean()) / (self.policy_returns.std() + 1e-8)
