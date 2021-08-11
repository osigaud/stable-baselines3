import math
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch as th
from gym import spaces

from stable_baselines3.common.buffers import BaseBuffer
from stable_baselines3.common.preprocessing import get_obs_shape
from stable_baselines3.common.type_aliases import ReplayBufferSamples, RolloutBufferSamples
from stable_baselines3.common.vec_env import VecNormalize


class EpisodicBuffer(BaseBuffer):
    """
    Episodic buffer used in on-policy PG algorithms like REINFORCE
    It corresponds to episodes collected using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: cpu or gpu
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    :param n_steps: N of N-step return
    :param gradient_name: name of the type of gradient
    :param nb_rollouts: Number of rollouts to fill the buffer
    :param max_episode_steps: Maximum length of an episode
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
        n_steps: int = 5,
        gradient_name: str = "discount",
        beta: float = 1.0,
        nb_rollouts: int = 1,
        max_episode_steps: int = 1,
    ):
        print(nb_rollouts)
        print(max_episode_steps)
        buffer_size = nb_rollouts * max_episode_steps

        super(EpisodicBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        self.gradient_name = gradient_name
        self.gae_lambda = gae_lambda
        self.n_steps = n_steps
        self.gamma = gamma
        self.beta = beta
        # maximum steps in episode
        self.max_episode_steps = max_episode_steps
        self.current_idx = 0
        self.episode_idx = 0
        # Counter to prevent overflow
        self.episode_steps = 0
        self.nb_rollouts = nb_rollouts

        # buffer with episodes

        # number of episodes which can be stored until buffer size is reached
        # self.nb_rollouts = self.buffer_size // self.max_episode_steps
        self.current_idx = 0
        # Counter to prevent overflow
        self.episode_steps = 0

        # Get shape of observation and goal (usually the same)
        self.obs_shape = get_obs_shape(self.observation_space)

        # episode length storage, needed for episodes which has less steps than the maximum length
        self.episode_lengths = np.zeros(self.nb_rollouts, dtype=np.int64)

        self.advantages = np.zeros((self.nb_rollouts, self.max_episode_steps), dtype=np.float32)
        self.returns = np.zeros((self.nb_rollouts, self.max_episode_steps), dtype=np.float32)
        self.values = np.zeros((self.nb_rollouts, self.max_episode_steps), dtype=np.float32)
        self.log_probs = np.zeros((self.nb_rollouts, self.max_episode_steps), dtype=np.float32)

        assert self.n_envs == 1, "Episode buffer only support single env for now"

        # input dimensions for buffer initialization
        # TODO: properly handle timeouts?
        self.input_shape = {
            "observation": (self.n_envs,) + self.obs_shape,
            "action": (self.action_dim,),
            "reward": (1,),
            # "next_obs": (self.n_envs,) + self.obs_shape,
            "done": (1,),
        }
        self._buffer = {
            key: np.zeros((self.nb_rollouts, self.max_episode_steps, *dim), dtype=np.float32)
            for key, dim in self.input_shape.items()
        }
        self.reset()

    def add(
        self,
        obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:

        self._buffer["observation"][self.episode_idx][self.current_idx] = obs
        self._buffer["action"][self.episode_idx][self.current_idx] = action
        self._buffer["done"][self.episode_idx][self.current_idx] = done
        self._buffer["reward"][self.episode_idx][self.current_idx] = reward

        # update current pointer
        self.current_idx += 1
        self.episode_steps += 1

        if done or self.episode_steps >= self.max_episode_steps:
            self.store_episode()
            self.episode_steps = 0

    def get_samples(self) -> RolloutBufferSamples:
        assert self.full, "digging into a non full batch"

        total_steps = sum(self.episode_lengths)
        all_transitions = np.concatenate([np.arange(ep_len) for ep_len in self.episode_lengths]).astype(np.uint64)
        all_episodes = np.concatenate([np.ones(ep_len) * ep_idx for ep_idx, ep_len in enumerate(self.episode_lengths)])
        all_episodes = all_episodes.astype(np.uint64)
        # Retrieve all transition and flatten the arrays
        return RolloutBufferSamples(
            self.to_torch(self._buffer["observation"][all_episodes, all_transitions].reshape(total_steps, *self.obs_shape)),
            self.to_torch(self._buffer["action"][all_episodes, all_transitions].reshape(total_steps, self.action_dim)),
            self.to_torch(self.values[all_episodes, all_transitions].reshape(total_steps)),
            self.to_torch(self.log_probs[all_episodes, all_transitions].reshape(total_steps)),
            self.to_torch(self.advantages[all_episodes, all_transitions].reshape(total_steps)),
            self.to_torch(self.returns[all_episodes, all_transitions].reshape(total_steps)),
        )

    def _get_samples(
        self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> Union[ReplayBufferSamples, RolloutBufferSamples]:
        """
        :param batch_inds:
        :param env:
        :return:
        """
        raise NotImplementedError()

    def store_episode(self) -> None:
        """
        Increment episode counter
        and reset transition pointer.
        """
        # add episode length to length storage
        self.episode_lengths[self.episode_idx] = self.current_idx

        self.episode_idx += 1
        if self.episode_idx == self.nb_rollouts:
            self.full = True
            self.episode_idx = 0
        # reset transition pointer
        self.current_idx = 0

    @property
    def n_episodes_stored(self) -> int:
        if self.full:
            return self.nb_rollouts
        return self.episode_idx

    def size(self) -> int:
        """
        :return: The current number of transitions in the buffer.
        """
        return int(np.sum(self.episode_lengths))

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.episode_idx = 0
        self.current_idx = 0
        self.full = False
        self.episode_lengths = np.zeros(self.nb_rollouts, dtype=np.int64)

    def post_processing(self, last_values: th.Tensor) -> None:
        """
        Post-processing step: compute the return using different gradient computation criteria
        For more information, see https://www.youtube.com/watch?v=GcJ9hl3T6x8&t=23s

        :param last_values: state value estimation for the last step (one for each env)
        """
        if self.gradient_name == "beta":
            self.exponentiate_rewards(self.beta)
        elif self.gradient_name == "sum":
            self.sum_rewards()
        elif self.gradient_name == "discount":
            self.discounted_sum_rewards()
        elif self.gradient_name == "normalize":
            self.normalize_rewards()
        elif self.gradient_name == "baseline":
            self.discounted_sum_rewards()
            self.subtract_baseline()
        elif self.gradient_name == "n_step":
            self.n_step_return()
        elif self.gradient_name == "gae":
            self.process_gae(last_values)
        else:
            raise NotImplementedError(f"The gradient {self.gradient_name} is not implemented")

    def process_gae(self, last_values: th.Tensor) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain vanilla advantage (A(s) = R - V(S))
        where R is the discounted reward with value bootstrap,
        set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        :param last_values: state value estimation for the last step (one for each env)

        """
        # Convert to numpy
        last_values = last_values.clone().cpu().numpy().flatten()

        last_gae_lam = 0
        for ep in range(self.nb_rollouts):
            for step in reversed(range(self.episode_lengths[ep])):
                if step == self.episode_lengths[ep]:
                    delta = self._buffer["reward"][ep][step] + self.gamma * last_values - self.values[ep][step]
                else:
                    delta = self._buffer["reward"][ep][step] + self.gamma * self.values[ep][step + 1] - self.values[ep][step]
                last_gae_lam = delta + self.gamma * self.gae_lambda * last_gae_lam
                self.advantages[ep][step] = last_gae_lam
            # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
            # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
            self.returns[ep] = self.advantages[ep] + self.values[ep]

    def discounted_sum_rewards(self) -> None:
        """
        Apply a discounted sum of rewards to all samples of all episodes
        :return: nothing
        """
        for ep in range(self.nb_rollouts):
            summ = 0
            for i in reversed(range(self.episode_lengths[ep])):
                summ = summ * self.gamma + self._buffer["reward"][ep][i]
                self.returns[ep][i] = summ

    def sum_rewards(self) -> None:
        """
        Apply a sum of rewards to all samples of all episodes
        :return: nothing
        """
        for ep in range(self.nb_rollouts):
            summ = np.sum(self._buffer["reward"][ep])
            for i in reversed(range(self.episode_lengths[ep])):
                self.returns[ep][i] = summ

    def subtract_baseline(self) -> None:
        """
        Subtracts the values to the reward of all samples of all episodes
        :return: nothing
        """
        for ep in range(self.nb_rollouts):
            for i in range(self.episode_lengths[ep]):
                self.returns[ep][i] = self._buffer["reward"][ep][i] - self.values[ep][i]

    def n_step_return(self) -> None:
        """
        Apply Bellman backup n-step return to all rewards of all samples of all episodes
        :return: nothing
        """
        for ep in range(self.nb_rollouts):
            for i in range(self.episode_lengths[ep]):
                horizon = i + self.n_steps
                summ = self._buffer["reward"][ep][i]
                if horizon < self.episode_lengths[ep]:
                    bootstrap_val = self.values[ep][horizon]
                    summ += self.gamma ** self.n_steps * bootstrap_val
                for j in range(1, self.n_steps):
                    if i + j >= self.episode_lengths[ep]:
                        break
                    summ += self.gamma ** j * self._buffer["reward"][ep][i + j]
                self.returns[ep][i] = summ

    def normalize_rewards(self) -> None:
        """
        Apply a normalized and discounted sum of rewards to all samples of all episodes
        """
        reward_mean = 0
        reward_pool = []
        self.sum_rewards()
        for ep in range(self.nb_rollouts):
            reward_pool += self.returns[ep][0]
        reward_std = np.std(reward_pool)
        if reward_std > 0:
            reward_mean = np.mean(reward_pool)
            # print("normalize_rewards : ", reward_std, "mean=", reward_mean)
        for ep in range(self.nb_rollouts):
            summ = 0
            for i in reversed(range(self.episode_lengths[ep])):
                summ = summ * self.gamma + self._buffer["reward"][ep][i]
                self.returns[ep][i] = (summ - reward_mean) / reward_std

    def normalize_discounted_rewards(self) -> None:
        """
        Apply a normalized and discounted sum of rewards to all samples of the episode
        """
        reward_mean = 0
        reward_pool = []
        self.discounted_sum_rewards()
        for ep in range(self.nb_rollouts):
            for i in range(self.episode_lengths[ep]):
                reward_pool += self.returns[ep][i]
        reward_std = np.std(reward_pool)
        if reward_std > 0:
            reward_mean = np.mean(reward_pool)
            # print("normalize_rewards : ", reward_std, "mean=", reward_mean)
        for ep in range(self.nb_rollouts):
            summ = 0
            for i in reversed(range(self.episode_lengths[ep])):
                summ = summ * self.gamma + self._buffer["reward"][ep][i]
                self.returns[ep][i] = (summ - reward_mean) / reward_std

    def exponentiate_rewards(self, beta) -> None:
        """
        Apply an exponentiation factor to the rewards of all samples of all episodes
        :param beta: the exponentiation factor
        """
        for ep in range(self.nb_rollouts):
            for i in range(self.episode_lengths[ep]):
                self.returns[ep][i] = math.exp(self._buffer["reward"][ep][i] / beta)
