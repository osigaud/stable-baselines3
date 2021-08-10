import time
from typing import List, Optional, Union
import gym
import numpy as np

from stable_baselines3.common.type_aliases import GymObs, GymStepReturn

class CustomMonitor(gym.Wrapper):
    """
    A monitor wrapper for Gym environments, it is used to know the episode reward, length, time and other data.
    :param env: The environment to be wrapped
    :param dir_name: the location to save a log file
    :param file_name: the location to save a log file, can be None for no log
    :param allow_early_resets: allows the reset of the environment before it is done
    """

    def __init__(
        self,
        env: gym.Env,
        dir_name: Optional[str] = None,
        file_name: Optional[str] = None,
        allow_early_resets: bool = True,
    ):
        super(CustomMonitor, self).__init__(env=env)
        self.t_start = time.time()
        if file_name is not None:
            self.results_writer = CustomResultsWriter(dir_name, file_name)
        else:
            self.results_writer = None
        self.allow_early_resets = allow_early_resets
        self.rewards = None
        self.needs_reset = True
        self.episode_returns = []
        self.episode_lengths = []
        self.episode_times = []
        self.total_steps = 0
        self.current_reset_info = {}  # extra info about the current episode, that was passed in during reset()
        self.num_episode = 0

    def reset(self, **kwargs) -> GymObs:
        """
        Calls the Gym environment reset. Can only be called if the environment is over, or if allow_early_resets is True
        :param kwargs: Extra keywords saved for the next episode. only if defined by reset_keywords
        :return: the first observation of the environment
        """
        if not self.allow_early_resets and not self.needs_reset:
            raise RuntimeError(
                "Tried to reset an environment before done. If you want to allow early resets, "
                "wrap your env with Monitor(env, path, allow_early_resets=True)"
            )
        self.rewards = []
        self.needs_reset = False
        return self.env.reset(**kwargs)

    def start_again(self) -> None:
        self.num_episode = 0

    def step(self, action: Union[np.ndarray, int]) -> GymStepReturn:
        """
        Step the environment with the given action
        :param action: the action
        :return: observation, reward, done, information
        """
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        observation, reward, done, info = self.env.step(action)
        self.rewards.append(reward)
        if done:
            self.num_episode += 1
            self.needs_reset = True
            ep_rew = sum(self.rewards)
            ep_len = len(self.rewards)
            ep_info = {round(ep_rew, 6), ep_len}
            self.episode_returns.append(ep_rew)
            self.episode_lengths.append(ep_len)
            self.episode_times.append(time.time() - self.t_start)
            ep_info.update(self.current_reset_info)
            if self.results_writer:
                self.results_writer.write_row(str(self.num_episode), str(round(ep_rew, 6)), str(ep_len))
        self.total_steps += 1
        return observation, reward, done, info

    def close(self) -> None:
        """
        Closes the environment
        """
        super(CustomMonitor, self).close()
        if self.results_writer is not None:
            self.results_writer.close()

    def get_total_steps(self) -> int:
        """
        Returns the total number of timesteps
        :return:
        """
        return self.total_steps

    def get_episode_rewards(self) -> List[float]:
        """
        Returns the rewards of all the episodes
        :return:
        """
        return self.episode_returns

    def get_episode_lengths(self) -> List[int]:
        """
        Returns the number of timesteps of all the episodes
        :return:
        """
        return self.episode_lengths

    def get_episode_times(self) -> List[float]:
        """
        Returns the runtime in seconds of all the episodes
        :return:
        """
        return self.episode_times


class CustomResultsWriter:
    """
    A result writer that saves the data from the `Monitor` class
    :param dir_name: the location to save a log file
    :param file_name: the root name of the log files
    """

    def __init__(
        self,
        dir_name: str = "",
        file_name: str = ""
    ):
        duration_name = dir_name + "duration_" + file_name + ".txt"
        self.duration_file = open(duration_name, "w")
        reward_name = dir_name + "reward_" + file_name + ".txt"
        self.reward_file = open(reward_name, "w")

    def write_row(self, num, reward, duration) -> None:
        """
        Write into the result files
        :param reward: Total reward of the episode
        :param duration: Duration of the episode
        """
        self.reward_file.write(num + ' ' + reward + '\n')
        self.duration_file.write(num + ' ' + duration + '\n')
        self.reward_file.flush()
        self.duration_file.flush()

    def close(self) -> None:
        """
        Close the files
        """
        self.reward_file.close()
        self.duration_file.close()
