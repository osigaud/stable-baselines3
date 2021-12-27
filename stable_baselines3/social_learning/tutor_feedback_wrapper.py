import gym


class TutorFeedbackWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    # This is a custom wrapper for playing with reward shaping in the MountainCarContinuous environment
    """

    def __init__(self, env):
        # Call the parent constructor, so we can access self.env later
        super(TutorFeedbackWrapper, self).__init__(env)
        self.nb_total_feedback = 0

    # This is the reward shaping function that you may want to tune better
    def tutor_reward_signal(self, obs, action, reward):
        if reward > 0.0:
            return reward
        if obs[1] < -0.01 and action[0] < 0:
            print("good left : ", obs[0], ":", obs[1], ":", action[0])
            self.nb_total_feedback += 1
            return abs(obs[1])
        elif -1.2 < obs[0] < -0.5 and obs[1] > 0.01 and action[0] > 0.5:
            print("good momentum : ", obs[0], ":", obs[1], ":", action[0])
            self.nb_total_feedback += 1
            return obs[1]
        elif obs[0] > -0.5 and obs[1] > 0.03 and action[0] > 0.5:
            print("good push right : ", obs[0], ":", obs[1], ":", action[0])
            self.nb_total_feedback += 1
            return obs[1]
        else:
            # print("no feedback")
            return reward

    def reset(self):
        """
        Reset the environment
        """
        obs = self.env.reset()
        print("reset")
        return obs

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
        """
        obs, reward, done, info = self.env.step(action)
        reward = self.tutor_reward_signal(obs, action, reward)
        return obs, reward, done, info
