import gym

# This is the reward shaping function that you may want to tune better
def tutor_reward_signal(obs, reward):
    if 0 < obs[0] < 0.5:
        return 10
    else:
        return reward


class TutorFeedbackWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    # This is a custom wrapper for playing with reward shaping in the MountainCarContinuous environment
    """

    def __init__(self, env):
        # Call the parent constructor, so we can access self.env later
        super(TutorFeedbackWrapper, self).__init__(env)

    def reset(self):
        """
        Reset the environment
        """
        obs = self.env.reset()
        return obs

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
        """
        obs, reward, done, info = self.env.step(action)
        reward = tutor_reward_signal(obs, reward)
        return obs, reward, done, info