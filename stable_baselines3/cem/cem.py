import os
from typing import Any, Dict, Optional, Type, Union

import numpy as np
import torch as th
from gym import spaces

from stable_baselines3.common import logger
from stable_baselines3.common.on_policy_algorithm import BaseAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule


class CEM(BaseAlgorithm):
    """

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param rms_prop_eps: RMSProp epsilon. It stabilizes square root computation in denominator
        of RMSProp update
    :param use_rms_prop: Whether to use RMSprop (default) or Adam as optimizer
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

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 100,
        tb_log_name: str = "run",
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "BaseAlgorithm":
        pass

    def _setup_model(self) -> None:
        pass

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 7e-4,
        rms_prop_eps: float = 1e-5,
        use_rms_prop: bool = True,
        normalize_advantage: bool = False,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):

        super(CEM, self).__init__(
            policy,
            env,
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

        self.normalize_advantage = normalize_advantage

        # Update optimizer inside the policy if we want to use RMSProp
        # (original implementation) rather than Adam
        if use_rms_prop and "optimizer_class" not in self.policy_kwargs:
            self.policy_kwargs["optimizer_class"] = th.optim.RMSprop
            self.policy_kwargs["optimizer_kwargs"] = dict(alpha=0.99, eps=rms_prop_eps, weight_decay=0)

        if _init_setup_model:
            self._setup_model()

    def train(self, params) -> None:
        """
        The main function to learn policies using the Cross Enthropy Method
         Update policy using the currently gathered
         rollout buffer (one gradient step over whole data).
        """
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)

        # Initialize variables
        self.list_weights = []
        self.best_weights = np.zeros(self.policy.get_weights_dim())
        self.list_rewards = np.zeros((int(params.nb_cycles)))
        self.best_reward = -1e38
        self.best_weights_idx = 0

        # Print the number of workers with the multi-thread
        # TODO : does multithreading work ?
        if params.multi_threading:
            workers = os.cpu_count()
            evals = int(params.nb_evals / workers)
            print("\n Multi-Threading Evals : " + str(workers) + " workers with each " + str(evals) + " evals to do")

        print("Shape of weights vector is: ", self.policy.get_weights_dim())

        if params.start_from_policy:
            # TODO(olivier): start from existing policy weights
            # starting_weights = get_starting_weights(pw)
            # centroid = starting_weights
            raise NotImplementedError()
        # Init the first centroid
        elif params.start_from_same_policy:
            centroid = self.policy.get_weights()
        else:
            centroid = np.random.rand(self.policy.get_weights_dim())

        self.policy.set_weights(centroid)
        initial_score = self.evaluate_episode(self.policy, params.deterministic_eval, params)
        self.list_weights.append(centroid)
        # Set the weights with this random centroid

        # Init the noise matrix
        noise = np.diag(np.ones(self.policy.get_weights_dim()) * params.sigma)
        # Init the covariance matrix
        var = np.diag(np.ones(self.policy.get_weights_dim()) * np.var(centroid)) + noise
        # var=np.diag(np.ones(self.policy.get_weights_dim())*params.lr_actor**2)+noise
        # Init the rng
        rng = np.random.default_rng()
        # Training Loop
        with SlowBar("Performing a repetition of CEM", max=params.nb_cycles) as bar:  # pytype: disable=name-error
            for cycle in range(params.nb_cycles):
                rewards = np.zeros(params.population)
                weights = rng.multivariate_normal(centroid, var, params.population)
                for p in range(params.population):
                    self.policy.set_weights(weights[p])
                    batch = self.make_monte_carlo_batch(params.nb_trajs, params.render, self.policy, True)
                    rewards[p] = batch.train_policy_cem(self.policy, params.bests_frac)

                elites_nb = int(params.elites_frac * params.population)
                elites_idxs = rewards.argsort()[-elites_nb:]
                elites_weights = [weights[i] for i in elites_idxs]
                # update the best weights
                centroid = np.array(elites_weights).mean(axis=0)
                var = np.cov(elites_weights, rowvar=False) + noise
                self.env.write_cov(cycle, np.linalg.norm(var))
                distance = np.linalg.norm(centroid - self.list_weights[-1])
                self.env.write_distances(cycle, distance)

                # policy evaluation part
                self.policy.set_weights(centroid)

                self.list_weights.append(self.policy.get_weights())
                self.write_angles_global(cycle)

                # policy evaluation part
                if (cycle % params.eval_freq) == 0:
                    total_reward = self.evaluate_episode(self.policy, params.deterministic_eval, params)
                    # write and store reward
                    self.env.write_reward(cycle + 1, total_reward)
                    self.list_rewards[cycle] = total_reward

                # Save best reward agent (no need for averaging if the policy is deterministic)
                if self.best_reward < total_reward:
                    self.best_reward = total_reward
                    self.best_weights = self.list_weights[-1]
                    self.best_weights_idx = cycle
                # Save the best policy obtained
                if (cycle % params.save_freq) == 0:
                    raise NotImplementedError()
                    # pw.save(method="CEM", cycle=cycle + 1, score=total_reward)
                bar.next()

        # pw.rename_best(method="CEM",best_cycle=self.best_weights_idx,best_score=self.best_reward)
        print("Best reward: ", self.best_reward)
        print("Best reward iter: ", self.best_weights_idx)
