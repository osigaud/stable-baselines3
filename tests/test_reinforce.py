import numpy as np
import pytest

from stable_baselines3 import REINFORCE


@pytest.mark.parametrize(
    "gradient_name",
    [
        "beta",
        "sum",
        "discount",
        "normalized sum",
        # "baseline",
        # "n_step",
        "gae",
    ],
)
# @pytest.mark.parametrize("nb_rollouts", [1, 3])
def test_reinforce(gradient_name):
    # Make numpy throw exceptions
    np.seterr(all="raise")
    kwargs = dict(beta=0.9) if gradient_name == "beta" else {}
    model = REINFORCE(
        "MlpPolicy",
        "CartPole-v1",
        gradient_name=gradient_name,
        seed=1,
        verbose=1,
        # TODO(olivier): re-add automatic max episode steps detection
        max_episode_steps=500,
        # nb_rollouts=nb_rollouts,
        **kwargs,
    )
    model.learn(nb_epochs=10, log_interval=2)
