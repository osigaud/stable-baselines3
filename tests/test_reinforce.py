import pytest

from stable_baselines3 import REINFORCE


@pytest.mark.parametrize(
    "gradient_name",
    [
        "beta",
        "sum",
        "discount",
        "normalize",
        "baseline",
        "n_step",
        "gae",
    ],
)
def test_reinforce(gradient_name):
    model = REINFORCE(
        "MlpPolicy",
        "CartPole-v1",
        gradient_name=gradient_name,
        seed=1,
        verbose=1,
    )
    model.learn(1000)
