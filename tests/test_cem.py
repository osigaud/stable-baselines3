import pytest

from stable_baselines3 import CEM


@pytest.mark.parametrize("env_id", ["CartPole-v1", "Pendulum-v0"])
def test_cem(env_id):
    model = CEM("MlpPolicy", env_id, verbose=1, pop_size=3, seed=1, policy_kwargs=dict(net_arch=[64]))
    model.learn(nb_iterations=3)
