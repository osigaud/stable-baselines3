from typing import Any, Dict, Optional, Type, Union

import numpy as np
import torch as th
from gym import spaces

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.on_policy_algorithm import BaseAlgorithm, BasePolicy
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule


def evaluate(model, env) -> float:
    fitness, _ = evaluate_policy(model, env)
    return fitness


def get_params_tmp(model) -> np.ndarray:
    mean_params = dict(
        (key, value)
        for key, value in model.state_dict().items()
        if ("policy" in key or "shared_net" in key or "action" in key)
    )
    params = model._params_to_vector(model.policy)
    print(mean_params.values())
    print(params)
    return params


class CEM(BaseAlgorithm):
    """
    The Cross Entropy Method
    The built-in policy corresponds to the centroid of the population at each generation

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
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
        pop_size: int = 20,
        elit_frac_size: float = 0.2,
        sigma: float = 0.2,
        noise_multiplier: float = 0.999,
        policy_base: Type[BasePolicy] = ActorCriticPolicy,
        learning_rate: Union[float, Schedule] = 7e-4,
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

        self.rng = np.random.default_rng()

        self.pop_size = pop_size
        self.elit_frac_size = elit_frac_size
        self.elites_nb = int(self.elit_frac_size * self.pop_size)

        self.sigma = sigma
        self.noise = None
        self.var = None
        self.d3rlpy_model = True

        self.noise_multiplier = noise_multiplier

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
        params = self.get_params()
        self.policy_dim = params.shape[0]
        print("policy dim:", self.policy_dim)

    @staticmethod
    def to_numpy(tensor):
        return tensor.detach().numpy().flatten()

    def get_params(self) -> np.ndarray:
        return self.policy.parameters_to_vector()  # note: also retrieves critic params...

    def set_params(self, indiv) -> None:
        self.policy.load_from_vector(indiv.copy())

    def update_noise(self) -> None:
        self.noise = self.noise * self.noise_multiplier

    def make_random_indiv(self) -> np.ndarray:
        return np.random.rand(self.policy.get_weights_dim())

    def init_var(self, centroid) -> None:
        self.noise = np.diag(np.ones(self.policy_dim) * self.sigma)
        self.var = np.diag(np.ones(self.policy_dim) * np.var(centroid)) + self.noise

    def one_loop(self, iter: int) -> None:
        centroid = self.get_params()
        self.update_noise()
        scores = np.zeros(self.pop_size)
        weights = self.rng.multivariate_normal(centroid, self.var, self.pop_size)

        for i in range(self.pop_size):
            self.set_params(weights[i])  # TODO: rather use a policy built on the fly
            scores[i] = evaluate(self.policy, self.env)

        elites_idxs = scores.argsort()[-self.elites_nb :]
        scores.sort()
        print("scores:", scores)
        self.logger.record("train/n_updates", iter, exclude="tensorboard")
        self.logger.record("train/best_score", scores[-1])
        elites_weights = [weights[i] for i in elites_idxs]
        # update the best weights
        centroid = np.array(elites_weights).mean(axis=0)
        self.var = np.cov(elites_weights, rowvar=False) + self.noise
        self.set_params(centroid)

    def train(self, nb_iterations: int, callback: BaseCallback) -> None:
        """
        The main function to learn policies using the Cross Entropy Method
        """
        # Init the covariance matrix
        centroid = self.get_params()
        self.init_var(centroid)

        for iteration in range(nb_iterations):
            callback.on_rollout_start()
            self.one_loop(iter)

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
        print(eval_env)
        total_steps, callback = self._setup_learn(
            total_steps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
        )
        callback.on_training_start(locals(), globals())
        training_ok = self.train(nb_iterations, callback=callback)
        if not training_ok:
            raise NotImplementedError("Collect rollout stopped unexpectedly")

        callback.on_training_end()
        return self
