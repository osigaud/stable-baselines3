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
        "normalized discounted",
        "n step",
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
        # nb_rollouts=nb_rollouts,
        **kwargs,
    )
    model.learn(50, log_interval=100)


@pytest.mark.parametrize(
    "critic_estim_method",
    [
        "mc",
        "td",
        "n steps",
    ],
)
@pytest.mark.parametrize("use_baseline", [False, True])
def test_critic(critic_estim_method, use_baseline):
    # Make numpy throw exceptions
    np.seterr(all="raise")
    model = REINFORCE(
        "MlpPolicy",
        "CartPole-v1",
        gradient_name="sum",
        seed=1,
        verbose=1,
        critic_estim_method=critic_estim_method,
        use_baseline=use_baseline,
        # **kwargs,
    )
    model.learn(50, log_interval=100)
