import gym

class TutorInstructionWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    # This is a custom wrapper for enforcing instructions in the MountainCarContinuous environment
    # We need to store the current observation to determine the instruction to give (as a function of the observation)
    """

    def __init__(self, env):
        # Call the parent constructor, so we can access self.env later
        super(TutorInstructionWrapper, self).__init__(env)
        self.current_obs = self.reset()
        self.nb_total_instructions = 0

    def reset(self):
        """
        Reset the environment
        """
        obs = self.env.reset()
        self.current_obs = obs
        return obs


    def tutor_instruction_signal(self, action):
        if -1.0 < self.current_obs[0] < 0 and self.current_obs[1] > 0.01:
            self.nb_total_instructions += 1
            return [1.0]
        elif -1.0 < self.current_obs[0] < -0.5 and self.current_obs[1] < -0.02:
            self.nb_total_instructions += 1
            return [-1.0]
        else:
            return action

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
        """
        action = self.tutor_instruction_signal(action)
        obs, reward, done, info = self.env.step(action)
        self.current_obs = obs
        return obs, reward, done, info