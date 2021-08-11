import pytest
import numpy as np

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
@pytest.mark.parametrize("nb_rollouts", [1, 3])
def test_reinforce(gradient_name, nb_rollouts):
    # Make numpy throw exceptions
    np.seterr(all="raise")
    kwargs = dict(beta=0.9) if gradient_name == "beta" else {}
    model = REINFORCE(
        "MlpPolicy",
        "CartPole-v1",
        gradient_name=gradient_name,
        seed=1,
        verbose=1,
        nb_rollouts=nb_rollouts,
        **kwargs,
    )
    model.learn(1000)
