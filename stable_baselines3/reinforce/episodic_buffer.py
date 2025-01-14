from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch as th
from gym import spaces

from stable_baselines3.common.buffers import BaseBuffer
from stable_baselines3.common.preprocessing import get_obs_shape
from stable_baselines3.common.type_aliases import EpisodicRolloutBufferSamples, ReplayBufferSamples, RolloutBufferSamples
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
        beta: float = 1.0,
        nb_rollouts: int = 1,
        max_episode_steps: int = 1,
        verbose=False,
    ):
        if verbose:
            print("nb rollouts:", nb_rollouts)
            print("max episode length:", max_episode_steps)
        buffer_size = nb_rollouts * max_episode_steps

        super(EpisodicBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

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
        print(self.obs_shape)

        # episode length storage, needed for episodes which has less steps than the maximum length
        self.episode_lengths = np.zeros(self.nb_rollouts, dtype=np.int64)

        assert self.n_envs == 1, "Episodic buffer only supports single env for now"

        self.reset()

    def add(
        self,
        obs: Dict[str, np.ndarray],
        action: np.ndarray,
        value: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:

        self._buffer["observation"][self.episode_idx, self.current_idx] = obs
        self._buffer["action"][self.episode_idx, self.current_idx] = action
        self.values[self.episode_idx, self.current_idx] = value
        self.rewards[self.episode_idx, self.current_idx] = reward
        self.episode_starts[self.episode_idx, self.current_idx] = episode_start
        self.dones[self.episode_idx, self.current_idx] = done
        # update current pointer
        self.current_idx += 1
        self.episode_steps += 1

        if done or self.episode_steps >= self.max_episode_steps:
            self.store_episode()
            self.episode_steps = 0

    def get_all_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve all samples valid indices, taking episode length
        into account.
        """
        all_episodes = np.concatenate([np.ones(ep_len) * ep_idx for ep_idx, ep_len in enumerate(self.episode_lengths)])
        all_transitions = np.concatenate([np.arange(ep_len) for ep_len in self.episode_lengths])
        return all_episodes.astype(np.uint64), all_transitions.astype(np.uint64)

    def get_samples(self) -> EpisodicRolloutBufferSamples:

        total_steps = sum(self.episode_lengths)
        all_indices = self.get_all_indices()
        # Retrieve all transition and flatten the arrays
        return EpisodicRolloutBufferSamples(
            self.to_torch(self._buffer["observation"][all_indices].reshape(total_steps, *self.obs_shape)),
            self.to_torch(self._buffer["action"][all_indices].reshape(total_steps, self.action_dim)),
            self.to_torch(self.policy_returns[all_indices].reshape(total_steps)),
            self.to_torch(self.target_values[all_indices].reshape(total_steps)),
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
        and reset current episode index.
        """
        # add episode length to length storage
        self.episode_lengths[self.episode_idx] = self.current_idx
        self.episode_idx += 1
        self.current_idx = 0

    @property
    def n_episodes_stored(self) -> int:
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
        self.values = np.zeros((self.nb_rollouts, self.max_episode_steps), dtype=np.float32)
        self.log_probs = np.zeros((self.nb_rollouts, self.max_episode_steps), dtype=np.float32)
        self.episode_starts = np.zeros((self.nb_rollouts, self.max_episode_steps), dtype=np.float32)
        self.dones = np.zeros((self.nb_rollouts, self.max_episode_steps), dtype=np.float32)
        # input dimensions for buffer initialization
        self.input_shape = {
            "observation": (self.n_envs,) + self.obs_shape,
            "action": (self.action_dim,),
        }
        self._buffer = {
            key: np.zeros((self.nb_rollouts, self.max_episode_steps, *dim), dtype=np.float32)
            for key, dim in self.input_shape.items()
        }
        self.policy_returns = np.zeros((self.nb_rollouts, self.max_episode_steps), dtype=np.float32)
        self.target_values = np.zeros((self.nb_rollouts, self.max_episode_steps), dtype=np.float32)
        self.rewards = np.zeros((self.nb_rollouts, self.max_episode_steps), dtype=np.float32)
        self.episode_idx = 0
        self.current_idx = 0
        self.episode_lengths = np.zeros(self.nb_rollouts, dtype=np.int64)

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
        for ep, ep_len in enumerate(self.episode_lengths):
            self.policy_returns[ep, :] = self.rewards[ep, :ep_len].sum()

    def get_normalized_rewards(self) -> None:
        """
        Normalize rewards of all samples of all episodes
        """
        all_rewards = self.rewards[self.get_all_indices()]
        self.policy_returns = (self.policy_returns - all_rewards.mean()) / (all_rewards.std() + 1e-8)

    def get_normalized_sum(self) -> None:
        """
        Normalize rewards of all samples of all episodes
        """
        self.get_sum_rewards()
        all_returns = self.policy_returns[self.get_all_indices()]
        self.policy_returns = (self.policy_returns - all_returns.mean()) / (all_returns.std() + 1e-8)

    def get_normalized_discounted_rewards(self) -> None:
        """
        Apply a normalized and discounted sum of rewards to all samples of the episode
        """
        self.get_discounted_sum_rewards()
        # Note(antonin): shall we normalize with all discounted returns
        # or with all rewards
        all_returns = self.policy_returns[self.get_all_indices()]
        self.policy_returns = (self.policy_returns - all_returns.mean()) / (all_returns.std() + 1e-8)

    def get_exponentiated_rewards(self, beta) -> None:
        """
        Apply an exponentiation factor to the rewards of all samples of all episodes
        :param beta: the exponentiation factor
        """
        # TODO(antonin): add a clip parameter to clip large values?
        self.policy_returns[:, :] = np.exp(self.rewards[:, :] / beta)

    def get_target_values_mc(self) -> None:
        """
        Warning: is only OK for V values
        """
        self.get_discounted_sum_rewards()
        self.target_values = self.policy_returns.copy()

    def get_target_values_td(self) -> None:
        """ """
        for ep in range(self.nb_rollouts):
            for step in reversed(range(self.episode_lengths[ep])):
                if step == self.episode_lengths[ep] - 1:
                    # Episodic setting: last step is always terminal
                    # and we are not handling timeout separately yet
                    target = self.rewards[ep, step]
                else:
                    target = self.rewards[ep, step] + self.gamma * self.values[ep, step + 1]
                self.target_values[ep, step] = target

    def get_target_values_nsteps(self) -> None:
        """
        Warning, assumes that values[ep] correspond to V-values
        """
        for ep in range(self.nb_rollouts):
            for step in reversed(range(self.episode_lengths[ep])):
                if step == self.episode_lengths[ep] - 1:
                    # Episodic setting: last step is always terminal
                    # and we are not handling timeout separately yet
                    summ = self.rewards[ep, step]
                else:
                    horizon = step + self.n_steps
                    summ = self.rewards[ep, step]
                    if horizon < self.episode_lengths[ep]:
                        bootstrap_val = self.values[ep, horizon]
                        summ += self.gamma**self.n_steps * bootstrap_val
                    for j in range(1, self.n_steps):
                        if step + j >= self.episode_lengths[ep]:
                            break
                        summ += self.gamma**j * self.rewards[ep, step + j]
                self.target_values[ep, step] = summ

    def get_n_step_return(self) -> None:
        """
        Apply Bellman backup n-step return to all rewards of all samples of all episodes
        Though this seems to work in practice, not sure it makes much sense
        """
        for ep in range(self.nb_rollouts):
            for i in range(self.episode_lengths[ep]):
                horizon = i + self.n_steps
                summ = self.rewards[ep, i]
                if horizon < self.episode_lengths[ep]:
                    bootstrap_val = self.values[ep, horizon]
                    summ += self.gamma**self.n_steps * bootstrap_val
                for j in range(1, self.n_steps):
                    if i + j >= self.episode_lengths[ep]:
                        break
                    summ += self.gamma**j * self.rewards[ep, i + j]
                self.policy_returns[ep, i] = summ

    def process_gae(self) -> None:
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
        """

        last_gae_lam = 0
        for ep in range(self.nb_rollouts):
            for step in reversed(range(self.episode_lengths[ep])):
                if step == self.episode_lengths[ep] - 1:
                    # delta = self.rewards[ep, step] + self.gamma * last_values - self.values[ep, step]
                    # Episodic setting: last step is always terminal
                    # and we are not handling timeout separately yet
                    delta = self.rewards[ep, step] - self.values[ep, step]
                else:
                    delta = self.rewards[ep, step] + self.gamma * self.values[ep, step + 1] - self.values[ep, step]
                last_gae_lam = delta + self.gamma * self.gae_lambda * last_gae_lam
                self.policy_returns[ep, step] = last_gae_lam
            # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
            # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
            self.target_values[ep] = self.policy_returns[ep] + self.values[ep]
