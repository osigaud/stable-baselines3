import time
from copy import deepcopy
from typing import Any, Dict, Optional, Type, Union

import numpy as np
import torch as th
from gym import spaces

from stable_baselines3.cem.policies import CEMPolicy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.on_policy_algorithm import BaseAlgorithm, BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule


class CEM(BaseAlgorithm):
    """
    The Cross Entropy Method
    The built-in policy corresponds to the centroid of the population at each generation

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
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
        policy: Union[str, Type[CEMPolicy]],
        env: Union[GymEnv, str],
        pop_size: int = 10,
        elit_frac_size: float = 0.2,
        sigma: float = 0.2,
        noise_multiplier: float = 0.999,
        n_eval_episodes: int = 5,
        nb_iterations: Optional[int] = None,
        policy_base: Type[BasePolicy] = CEMPolicy,
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
            policy_base=policy_base,
            learning_rate=0.0,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
            ),
        )

        self.pop_size = pop_size
        self.elit_frac_size = elit_frac_size
        self.elites_nb = int(self.elit_frac_size * self.pop_size)
        self.n_eval_episodes = n_eval_episodes
        self.train_policy = None
        self.nb_iterations = nb_iterations
        self.best_score = -np.inf

        self.sigma = sigma
        self.noise = None
        self.var = None
        self.rng = None

        self.noise_multiplier = noise_multiplier

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self.set_random_seed(self.seed)
        self.rng = np.random.default_rng(self.seed)

        self.policy = self.policy_class(  # pytype:disable=not-instantiable
            self.observation_space, self.action_space, **self.policy_kwargs  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)
        params = self.get_params(self.policy)
        self.train_policy = deepcopy(self.policy)
        self.policy_dim = params.shape[0]
        print(f"policy dim: {self.policy_dim}")

    @staticmethod
    def to_numpy(tensor):
        return tensor.detach().numpy().flatten()

    def get_params(self, policy: CEMPolicy) -> np.ndarray:
        return policy.parameters_to_vector()  # note: also retrieves critic params...

    def set_params(self, policy: CEMPolicy, indiv: np.ndarray) -> None:
        policy.load_from_vector(indiv.copy())

    def update_noise(self) -> None:
        self.noise = self.noise * self.noise_multiplier

    def make_random_indiv(self) -> np.ndarray:
        return np.random.rand(self.policy.get_weights_dim())

    def init_var(self, centroid) -> None:
        self.noise = np.diag(np.ones(self.policy_dim) * self.sigma)
        self.var = np.diag(np.ones(self.policy_dim) * np.var(centroid)) + self.noise

    def one_loop(self, n_iter: int) -> None:
        centroid = self.get_params(self.train_policy)
        self.update_noise()
        scores = np.zeros(self.pop_size)
        weights = self.rng.multivariate_normal(centroid, self.var, self.pop_size)
        # Optional: Reset best at every iteration
        # self.best_score = -np.inf

        for i in range(self.pop_size):
            self.set_params(self.train_policy, weights[i])  # TODO: rather use a policy built on the fly
            episode_rewards, episode_lengths = evaluate_policy(
                self.train_policy, self.env, n_eval_episodes=self.n_eval_episodes, return_episode_rewards=True
            )
            scores[i] = np.mean(episode_rewards)
            # Save best params
            if scores[i] > self.best_score:
                self.best_score = scores[i]
                self.set_params(self.policy, weights[i])
            self.num_timesteps += sum(episode_lengths)
            print(f"indiv: {i} score {scores[i]:.2f}")

        elites_idxs = scores.argsort()[-self.elites_nb :]
        scores.sort()
        print("scores:", scores)
        self.logger.record("train/n_updates", n_iter, exclude="tensorboard")
        self.logger.record("train/mean_score", np.mean(scores))
        self.logger.record("train/best_score", scores[-1])
        self._dump_logs()

        elites_weights = [weights[i] for i in elites_idxs]
        # update the best weights
        centroid = np.array(elites_weights).mean(axis=0)
        self.var = np.cov(elites_weights, rowvar=False) + self.noise
        self.set_params(self.train_policy, centroid)

    def _dump_logs(self) -> None:
        """
        Write log.
        """
        time_elapsed = time.time() - self.start_time
        fps = int(self.num_timesteps / (time_elapsed + 1e-8))
        # self.logger.record("time/episodes", self._episode_num, exclude="tensorboard")
        # if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
        #     self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
        #     self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total timesteps", self.num_timesteps, exclude="tensorboard")

        # Pass the number of timesteps for tensorboard
        self.logger.dump(step=self.num_timesteps)

    def train(self, nb_iterations: int, callback: BaseCallback) -> bool:
        """
        The main function to learn policies using the Cross Entropy Method
        """
        # Init the covariance matrix
        centroid = self.get_params(self.train_policy)
        self.init_var(centroid)

        for iteration in range(1, nb_iterations + 1):
            callback.on_rollout_start()
            self.one_loop(iteration)

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            callback.on_rollout_end()
        return True

    def learn(
        self,
        total_timesteps: int = 100,
        nb_iterations: int = 100,
        callback: MaybeCallback = None,
        log_interval: int = 100,
        tb_log_name: str = "run",
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "BaseAlgorithm":

        total_steps = total_timesteps
        total_steps, callback = self._setup_learn(
            total_steps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
        )
        nb_iterations = self.nb_iterations or nb_iterations
        callback.on_training_start(locals(), globals())
        training_ok = self.train(nb_iterations, callback=callback)
        # Can be stopped early with a callback
        # if not training_ok:
        #     raise NotImplementedError("Collect rollout stopped unexpectedly")

        callback.on_training_end()
        return self
