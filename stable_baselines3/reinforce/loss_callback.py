from stable_baselines3.common.callbacks import BaseCallback
from typing import Any, Dict, List, Optional, TextIO, Tuple, Type, Union

class LossCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """

    def __init__(
            self,
            dir_name: Optional[str] = None,
            file_name: Optional[str] = None,
            verbose=0
    ):
        super(LossCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        policy_loss_name = dir_name + "policy_loss_" + file_name + ".txt"
        self.policy_loss_file = open(policy_loss_name, "w")
        critic_loss_name = dir_name + "critic_loss_" + file_name + ".txt"
        self.critic_loss_file = open(critic_loss_name, "w")

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        num = self.logger.name_to_value["time/episode"]
        policy_loss = self.logger.name_to_value["train/policy_loss"]
        critic_loss = self.logger.name_to_value["train/value_loss"]
        self.policy_loss_file.write(str(num) + " " + str(policy_loss) + "\n")
        self.critic_loss_file.write(str(num) + " " + str(critic_loss) + "\n")
        self.policy_loss_file.flush()
        self.critic_loss_file.flush()

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass
