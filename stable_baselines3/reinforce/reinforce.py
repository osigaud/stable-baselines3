import time
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
from torch.nn import functional as func

from gym import spaces

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.reinforce.episodic_buffer import EpisodicBuffer
from stable_baselines3.reinforce.expert_policies import continuous_mountain_car_expert_policy


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
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param rms_prop_eps: RMSProp epsilon. It stabilizes square root computation in denominator
        of RMSProp update
    :param normalize_advantage: Whether to normalize or not the advantage
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
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        gradient_name: str = "sum",
        beta: float = 0.0,
        max_episode_steps: Optional[int] = None,
        learning_rate: Union[float, Schedule] = 0.01,
        n_steps: int = 5,
        gamma: float = 0.99,
        gae_lambda: float = 1.0,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        optimizer_name: str = "adam",
        rms_prop_eps: float = 1e-5,
        normalize_advantage: bool = False,
        policy_base: Type[BasePolicy] = ActorCriticPolicy,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        substract_baseline: bool = False,
        uses_entropy: bool = False,
        critic_estim_method: str = "td"
    ):
        super(REINFORCE, self).__init__(
            policy,
            env,
            learning_rate=learning_rate,
            policy_base=policy_base,
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
        self.gradient_name = gradient_name
        self.beta = beta
        self.max_episode_steps = max_episode_steps
        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.normalize_advantage = normalize_advantage
        self.clip_grad = False
        self.substract_baseline = substract_baseline
        self.uses_entropy = uses_entropy
        self.episode_num = 0
        self.rollout_buffer = None
        self.critic_estim_method = critic_estim_method

        # Update optimizer inside the policy if we want to use RMSProp
        # (original implementation) rather than Adam
        if "optimizer_class" not in self.policy_kwargs:
            if optimizer_name == "rmsprop":
                self.policy_kwargs["optimizer_class"] = th.optim.RMSprop
                self.policy_kwargs["optimizer_kwargs"] = dict(alpha=0.99, eps=rms_prop_eps, weight_decay=0)
            elif optimizer_name == "sgd":
                self.policy_kwargs["optimizer_class"] = th.optim.SGD
                # TODO: missing kwargs?

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        self.policy = self.policy_class(  # pytype:disable=not-instantiable
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=False,
            **self.policy_kwargs  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]
        return state_dicts, []

    def init_buffer(self, nb_rollouts):
        self.rollout_buffer = EpisodicBuffer(
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
        assert self._last_obs is not None, "No previous observation was provided"
        n_steps = 0
        rollout_buffer.reset()
        callback.on_rollout_start()

        while rollout_buffer.n_episodes_stored < rollout_buffer.nb_rollouts:
            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                if not expert_pol:
                    actions, _, _ = self.policy.forward(obs_tensor)
                else:
                    actions = continuous_mountain_car_expert_policy(rollout_buffer.episode_steps, var=True)
            if not expert_pol:
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
            rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts, dones, infos)
            new_episode_idx = rollout_buffer.n_episodes_stored
            if new_episode_idx > old_episode_idx:
                self.episode_num += 1
                # self.logger.record("time/collect episode", self.episode_num)
            self._last_obs = new_obs
            self._last_episode_starts = dones

        callback.on_rollout_end()
        return True

    def compute_critic(self):
        """
        The method assumes a rollout has already been collected, the rollout buffer is ready
        """
        if self.critic_estim_method == "mc":
            self.rollout_buffer.get_target_values_mc()
        elif self.critic_estim_method == "td":
            self.rollout_buffer.get_target_values_td()
        elif self.critic_estim_method == "n steps":
            self.rollout_buffer.get_target_values_nsteps()
        else:
            raise NotImplementedError(f"The critic computation method {self.critic_estim_method} is unknown")

        # Get all data in one go
        rollout_data = self.rollout_buffer.get_samples()
        obs = rollout_data.observations
        actions = rollout_data.actions
        if isinstance(self.action_space, spaces.Discrete):
            # Convert discrete action from float to long
            actions = actions.long().flatten()
        # TODO: avoid second computation of everything because of the gradient
        values, log_prob, entropy = self.policy.evaluate_actions(obs, actions)
        target_values = rollout_data.returns
        values = values.flatten()
        value_loss = func.mse_loss(target_values, values)
        # print("value loss", value_loss)

        # Entropy loss favors exploration
        if self.uses_entropy:
            if entropy is None:
                # Approximate entropy when no analytical form
                entropy_loss = -th.mean(-log_prob)
            else:
                entropy_loss = -th.mean(entropy)
            self.logger.record("train/entropy_loss", entropy_loss.item())

            total_loss = self.ent_coef * entropy_loss + self.vf_coef * value_loss
        else:
            total_loss = value_loss
        self.logger.record("train/value_loss", value_loss.item())

        # Optimization step
        self.policy.optimizer.zero_grad()
        total_loss.backward()

        if self.clip_grad:
            # Clip grad norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy.optimizer.step()

    def regress_policy(self):
        """

        """
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

    def post_processing(self) -> None:
        """
        Post-processing step: compute the return using different gradient computation criteria
        For more information, see https://www.youtube.com/watch?v=GcJ9hl3T6x8&t=23s
        """
        if self.substract_baseline:
            self.compute_baseline_td()

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

    def train(self) -> None:
        """
        Update policy using the currently gathered
        rollout buffer (one gradient step over whole data).
        """
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        self.compute_critic()
        rollout_data = self.rollout_buffer.get_samples()
        obs = rollout_data.observations
        actions = rollout_data.actions
        if isinstance(self.action_space, spaces.Discrete):
            # Convert discrete action from float to long
            actions = actions.long().flatten()

        advantages = rollout_data.advantages
        # TODO: avoid second computation of everything because of the gradient
        _, log_prob, _ = self.policy.evaluate_actions(obs, actions)

        # Policy gradient loss
        policy_loss = -(advantages * log_prob).mean()
        total_loss = policy_loss
        # print("total loss", total_loss)

        # Optimization step
        self.policy.optimizer.zero_grad()
        total_loss.backward()

        if self.clip_grad:
            # Clip grad norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy.optimizer.step()

        self._n_updates += 1
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/policy_loss", policy_loss.item())
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

    def learn(
        self,
        nb_epochs: int,
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

        total_steps = nb_rollouts * self.max_episode_steps
        total_steps, callback = self._setup_learn(
            total_steps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
        )
        self.init_buffer(nb_rollouts)
        callback.on_training_start(locals(), globals())
        for i in range(nb_epochs):
            self.logger.record("time/iterations", i, exclude="tensorboard")
            self.learn_one_epoch(total_steps, callback)
        callback.on_training_end()
        return self

    def learn_one_epoch(
        self,
        total_time_steps: int,
        callback: MaybeCallback = None,
        expert_pol: bool = False,
    ) -> None:

        collect_ok = self.collect_rollouts(self.env, callback, self.rollout_buffer, expert_pol)
        if not collect_ok:
            raise NotImplementedError("Collect rollout stopped unexpectedly")

        self.post_processing()
        self._update_current_progress_remaining(self.num_timesteps, total_time_steps)
        # Display training infos
        fps = int(self.num_timesteps / (time.time() - self.start_time))

        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
        self.logger.record("time/total_time_steps", self.num_timesteps, exclude="tensorboard")
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
        if not collect_ok:
            raise NotImplementedError("Collect rollout stopped unexpectedly")
        self.regress_policy()

    def old_train(self) -> None:
        """
        Update policy using the currently gathered
        rollout buffer (one gradient step over whole data).
        """
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)

        # Get all data in one go
        rollout_data = self.rollout_buffer.get_samples()

        obs = rollout_data.observations
        actions = rollout_data.actions
        if isinstance(self.action_space, spaces.Discrete):
            # Convert discrete action from float to long
            actions = actions.long().flatten()

        advantages = rollout_data.advantages
        # if self.gradient_name == "normalized sum" or self.gradient_name == "normalized discounted":
            # print("advantages", advantages.shape, advantages)
        target_values = rollout_data.returns
        # TODO: avoid second computation of everything because of the gradient
        values, log_prob, entropy = self.policy.evaluate_actions(obs, actions)
        values = values.flatten()

        # Policy gradient loss
        policy_loss = -(advantages * log_prob).mean()
        # print("policy loss", policy_loss)

        if self.gradient_name == "baseline" or self.gradient_name == "gae" or self.gradient_name == "n step":
            # Value loss using the TD(gae_lambda) target
            value_loss = func.mse_loss(target_values, values)
            # print("value loss", value_loss)

            # Entropy loss favors exploration
            if entropy is None:
                # Approximate entropy when no analytical form
                entropy_loss = -th.mean(-log_prob)
            else:
                entropy_loss = -th.mean(entropy)

            compound_value_loss = self.ent_coef * entropy_loss + self.vf_coef * value_loss
            total_loss = policy_loss + compound_value_loss
            self.logger.record("train/value_loss", value_loss.item())
            self.logger.record("train/entropy_loss", entropy_loss.item())
        else:
            total_loss = policy_loss
        # print("total loss", total_loss)

        # Optimization step
        self.policy.optimizer.zero_grad()
        total_loss.backward()

        if self.clip_grad:
            # Clip grad norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy.optimizer.step()

        self._n_updates += 1
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/policy_loss", policy_loss.item())
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())