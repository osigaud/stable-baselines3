import gym
import numpy as np

from gym import GoalEnv, spaces

epsilon = 0.02

class CustomGoalEnv(GoalEnv):

    def __init__(self, env_name):
        # Call the parent constructor, so we can access self.env later
        self.env = gym.make(env_name)
        super(CustomGoalEnv, self).__init__()
        print (self.env.observation_space, self.env.action_space)
        self.observation_space = spaces.Dict(
                {
                    "observation": self.env.observation_space,
                    "achieved_goal": self.env.observation_space,
                    "desired_goal": self.env.observation_space,
                })
        self.action_space =  self.env.action_space
        # self.obs_space = self.env.obs_space

    def reset(self):
        """
        Reset the environment
        """
        self.goal = self.env.reset()
        obs = self.env.reset()
        return obs

    def compute_reward(self, obs, goal, info):
        dist = np.linalg.norm(goal - obs)
        if dist < epsilon:
            return 1
        else:
            return 0

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
        """
        obs, reward, done, info = self.env.step(action)
        assert reward == self.compute_reward(obs, self.goal, info)
        reward = self.compute_reward(obs, self.goal, info)

        return obs, reward, done, info