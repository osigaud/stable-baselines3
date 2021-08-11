from gym.envs.registration import register

register(
    id="CartPoleContinuous-v0",
    entry_point="my_gym.envs:ContinuousCartPoleEnv",
    max_episode_steps=999,
)
