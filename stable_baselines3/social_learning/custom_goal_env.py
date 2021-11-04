import gym
import numpy as np
import random

from gym import GoalEnv, spaces

epsilon = 0.02

class CustomGoalEnv(GoalEnv):

    def __init__(self, env_name, random_goal=True):
        # Call the parent constructor, so we can access self.env later
        self.env = gym.make(env_name)
        super(CustomGoalEnv, self).__init__()
        self.observation_space = spaces.Dict(
                 {
                     "observation": self.env.observation_space,
                     "achieved_goal": self.env.observation_space,
                     "desired_goal": self.env.observation_space,
                 })
        self.action_space =  self.env.action_space
        self.random_goal = random_goal

    def reset(self):
        """
        Reset the environment
        """
        # We draw a random desired goal
        # self.goal = self.env.reset()
        pos = random.uniform(-1.2, 0.5)
        vel = random.uniform(-0.02, 0.02)
        self.goal = np.array([pos, vel])
        # And a random state
        o = self.env.reset()
        obs = {
                    "observation": o,
                    "achieved_goal": o,
                    "desired_goal": self.goal,
                }
        return obs

    def compute_reward(self, obs, goal, info):
        # Sparse goal-conditioned reward: we are rewarded if we are close enough to the goal
        if self.random_goal:
            dist = np.linalg.norm(goal - obs)
            if dist < epsilon:
                print("reached random goal:", self.goal)
                return 1
            else:
                return 0
        else:
            reward = info
            return reward

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
        """
        o, reward, done, info = self.env.step(action)
        obs = {
            "observation": o,
            "achieved_goal": o,
            "desired_goal": self.goal,
        }
        info2 =  reward
        if self.random_goal:
            reward = self.compute_reward(o, self.goal, info2)
            dist = np.linalg.norm(self.goal - o)
            if dist < epsilon:
                done = True
            else:
                done = False
        return obs, reward, done, info