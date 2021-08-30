import time
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
from gym import spaces
from torch.nn import functional as func
from torch.utils.data import DataLoader, TensorDataset

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.her.her_replay_buffer import get_time_limit
from stable_baselines3.reinforce.episodic_buffer import EpisodicBuffer
from stable_baselines3.reinforce.expert_policies import continuous_mountain_car_expert_policy
from stable_baselines3.reinforce.policies import REINFORCEPolicy


class REINFORCE(BaseAlgorithm):
    """
    REINFORCE

    This is generic code for Basic Policy Gradient algorithms, among which REINFORCE is just an instance
    Paper: https://link.springer.com/content/pdf/10.1007/BF00992696.pdf
    Code: This implementation is an adaptation to SB3 of code from https://github.com/osigaud/Basic-Policy-Gradient-Labs

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: N of N-step return
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        policy: Union[str, Type[REINFORCEPolicy]],
        env: Union[GymEnv, str],
        gradient_name: str = "sum",
        beta: float = 0.0,
        max_episode_steps: Optional[int] = None,
        policy_base: Type[REINFORCEPolicy] = REINFORCEPolicy,
        learning_rate: Union[float, Schedule] = 0.01,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        device: Union[th.device, str] = "auto",
        create_eval_env: bool = False,
        seed: Optional[int] = None,
        _init_setup_model: bool = True,
        n_steps: int = 5,
        gamma: float = 0.99,
        gae_lambda: float = 0.98,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        critic_estim_method: Optional[str] = "mc",
        n_critic_epochs: int = 25,
        critic_batch_size: int = -1,  # complete batch
        buffer_class: Type[EpisodicBuffer] = EpisodicBuffer,
    ):
        super(REINFORCE, self).__init__(
            policy,
            env,
            policy_base=policy_base,
            learning_rate=learning_rate,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )
        self.beta = beta
        self.max_episode_steps = max_episode_steps
        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.episode_num = 0
        self.rollout_buffer = None
        self.log_interval = 10
        self.gradient_name = gradient_name
        self.critic_estim_method = critic_estim_method
        self.n_critic_epochs = n_critic_epochs
        self.critic_batch_size = critic_batch_size
        self.buffer_class = buffer_class

        if gradient_name == "gae":
            assert critic_estim_method is not None, "You must specify a critic estimation method when using GAE"

        # Retrieve max episode step automatically
        if self.env is not None:
            self.max_episode_steps = get_time_limit(self.env, max_episode_steps)

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        self.policy = self.policy_class(  # pytype:disable=not-instantiable
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            **self.policy_kwargs,  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)
        # Aliases
        self.actor = self.policy.actor
        self.critic = self.policy.critic

    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params() + ["actor", "critic"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        return state_dicts, []

    def init_buffer(self, nb_rollouts):
        self.rollout_buffer = self.buffer_class(
            self.observation_space,
            self.action_space,
            self.device,
            self.gae_lambda,
            self.gamma,
            self.n_envs,
            self.n_steps,
            self.beta,
            nb_rollouts,
            max_episode_steps=self.max_episode_steps,
        )

    def reset_episodes(self):
        self.episode_num = 0

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: EpisodicBuffer,
        expert_pol: bool = False,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param expert_pol: Whether uses an expert policy or the learned one
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)
        assert self._last_obs is not None, "No previous observation was provided"
        n_steps = 0
        rollout_buffer.reset()
        callback.on_rollout_start()

        while rollout_buffer.n_episodes_stored < rollout_buffer.nb_rollouts:
            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                if expert_pol:
                    actions = continuous_mountain_car_expert_policy(rollout_buffer.episode_steps, add_noise=True)
                else:
                    actions, log_probs = self.actor.forward(obs_tensor)
                    # Note(antonin): value computation is probably not needed anymore here
                    values = self.critic.forward(obs_tensor)
                    actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            old_episode_idx = rollout_buffer.n_episodes_stored
            rollout_buffer.add(
                obs=self._last_obs,
                action=actions,
                value=values,
                reward=rewards,
                episode_start=self._last_episode_starts,
                done=dones,
                infos=infos,
            )
            new_episode_idx = rollout_buffer.n_episodes_stored
            if new_episode_idx > old_episode_idx:
                self.episode_num += 1
            self._last_obs = new_obs
            self._last_episode_starts = dones

        callback.on_rollout_end()
        return True

    def regress_policy(self):
        """ """
        rollout_data = self.rollout_buffer.get_samples()
        obs = rollout_data.observations
        actions = rollout_data.actions
        action_loss = 1e20
        while action_loss > 0.1:
            self_actions, _, _ = self.policy.forward(obs)
            action_loss = func.mse_loss(actions, self_actions)
            self.policy.optimizer.zero_grad()
            action_loss.sum().backward()
            self.policy.optimizer.step()

    def compute_policy_returns(self) -> None:
        """
        Post-processing step: compute the return using different gradient computation criteria
        For more information, see https://www.youtube.com/watch?v=GcJ9hl3T6x8&t=23s
        """

        if self.gradient_name == "beta":
            self.rollout_buffer.get_exponentiated_rewards(self.beta)
        elif self.gradient_name == "sum":
            self.rollout_buffer.get_sum_rewards()
        elif self.gradient_name == "discount":
            self.rollout_buffer.get_discounted_sum_rewards()
        elif self.gradient_name == "normalized sum":
            self.rollout_buffer.get_normalized_sum()
        elif self.gradient_name == "normalized discounted":
            self.rollout_buffer.get_normalized_discounted_rewards()
        elif self.gradient_name == "n step":
            self.rollout_buffer.get_n_step_return()
        elif self.gradient_name == "gae":
            self.rollout_buffer.process_gae()
        else:
            raise NotImplementedError(f"The gradient {self.gradient_name} is not implemented")

    def update_critic(self):
        """
        The method assumes a rollout has already been collected, the rollout buffer is ready
        """
        if self.critic_estim_method == "mc":
            self.rollout_buffer.get_target_values_mc()
        elif self.critic_estim_method == "td":
            self.rollout_buffer.get_target_values_td()
        elif self.critic_estim_method == "n steps":
            self.rollout_buffer.get_target_values_nsteps()
        elif self.critic_estim_method == "gae":
            # TD(lambda) return are computed when computing GAE(lambda)
            self.rollout_buffer.process_gae()
        else:
            raise NotImplementedError(f"The critic computation method {self.critic_estim_method} is unknown")

        # Make the value function converge
        rollout_data = self.rollout_buffer.get_samples()
        batch_size = self.critic_batch_size
        if self.critic_batch_size == -1:
            batch_size = len(rollout_data.target_values)  # complete batch

        training_data = TensorDataset(rollout_data.observations, rollout_data.target_values)
        train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=False)

        for _ in range(self.n_critic_epochs):
            # NOTE: Optionally, we could recompute target (when using TD target)
            for observations, target_values in train_dataloader:
                values = self.critic(observations).flatten()

                value_loss = func.mse_loss(target_values, values)
                # Optimization step
                self.critic.optimizer.zero_grad()
                value_loss.backward()

                self.critic.optimizer.step()

        with th.no_grad():
            values = self.critic(rollout_data.observations).flatten().detach().cpu().numpy()

        explained_var = explained_variance(values, rollout_data.target_values.cpu().numpy())

        self.logger.record("train/value_loss", value_loss.item())
        self.logger.record("train/explained_variance", explained_var)

    def update_actor(self):
        """
        Update of the actor from the samples collected in collect_rollout, after post_processing them
        """
        rollout_data = self.rollout_buffer.get_samples()
        policy_returns = rollout_data.policy_returns

        with th.no_grad():
            values = self.critic(rollout_data.observations).flatten()

        actions = rollout_data.actions
        if isinstance(self.action_space, spaces.Discrete):
            # Convert discrete action from float to long
            actions = actions.long().flatten()

        log_prob, _ = self.actor.evaluate_actions(rollout_data.observations, actions)

        if self.critic_estim_method is not None and self.gradient_name != "gae":
            policy_returns -= values
        policy_loss = -(policy_returns * log_prob).mean()

        # Optimization step
        self.actor.optimizer.zero_grad()
        policy_loss.backward()
        self.actor.optimizer.step()
        self.logger.record("train/policy_loss", policy_loss.item())

    def train(self) -> None:
        """
        Update policy using the currently gathered
        rollout buffer (one gradient step over whole data).
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        if self.critic_estim_method is not None or self.gradient_name == "gae":
            self.update_critic()

        self.compute_policy_returns()

        self.update_actor()

        self._n_updates += 1
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if hasattr(self.actor, "log_std"):
            self.logger.record("train/std", th.exp(self.actor.log_std).mean().item())

    def learn_one_epoch(
        self,
        total_steps: int,
        callback: MaybeCallback = None,
        expert_pol: bool = False,
    ) -> None:

        collect_ok = self.collect_rollouts(self.env, callback, self.rollout_buffer, expert_pol)
        assert collect_ok, "Collect rollout stopped unexpectedly"

        self._update_current_progress_remaining(self.num_timesteps, total_steps)
        # Display training infos
        fps = int(self.num_timesteps / (time.time() - self.start_time))

        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        if self.logger.name_to_value["time/iterations"] % self.log_interval == 0:
            self.logger.dump(step=self.num_timesteps)
        self.train()

    def collect_expert_rollout(
        self,
        nb_rollouts: int,
        callback: MaybeCallback = None,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "REINFORCE",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> None:

        total_steps = nb_rollouts * self.max_episode_steps
        total_steps, _ = self._setup_learn(
            total_steps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
        )
        self.init_buffer(nb_rollouts)
        collect_ok = self.collect_rollouts(self.env, callback, self.rollout_buffer, expert_pol=True)
        assert collect_ok, "Collect rollout stopped unexpectedly"
        self.regress_policy()

    def learn(
        self,
        total_timesteps: Optional[int] = None,
        nb_epochs: Optional[int] = None,
        nb_rollouts: int = 1,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "REINFORCE",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        expert_pol: bool = False,
    ) -> "BaseAlgorithm":

        assert (
            total_timesteps is not None or nb_epochs is not None
        ), "You must specify either a total number of time steps or a number of epochs"
        if total_timesteps is None:
            total_steps = nb_rollouts * self.max_episode_steps
        else:
            total_steps = total_timesteps
        total_steps, callback = self._setup_learn(
            total_steps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
        )
        self.init_buffer(nb_rollouts)
        callback.on_training_start(locals(), globals())
        self.log_interval = log_interval

        if total_timesteps is None:
            for i in range(nb_epochs):
                self.logger.record("time/iterations", i, exclude="tensorboard")
                self.learn_one_epoch(total_steps, callback)
        else:
            i = 0
            while self.num_timesteps < total_timesteps:
                i += 1
                self.logger.record("time/iterations", i, exclude="tensorboard")
                self.learn_one_epoch(total_steps, callback)

        callback.on_training_end()
        return self
