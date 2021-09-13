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
        "normalized discount",
        "n step",
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
        critic_estim_method=None if gradient_name != "gae" else "mc",
        **kwargs,
    )
    model.learn(200, nb_rollouts=nb_rollouts, log_interval=2)


@pytest.mark.parametrize(
    "critic_estim_method",
    [
        "mc",
        "td",
        "n steps",
    ],
)
def test_critic(critic_estim_method):
    # Make numpy throw exceptions
    np.seterr(all="raise")
    model = REINFORCE(
        "MlpPolicy",
        "CartPole-v1",
        gradient_name="sum",
        seed=1,
        verbose=1,
        critic_estim_method=critic_estim_method,
    )
    model.learn(50, log_interval=2)


def test_expert_mountaincar():
    model = REINFORCE(
        "MlpPolicy",
        "MountainCarContinuous-v0",
        gradient_name="discount",
        seed=1,
        verbose=1,
        critic_estim_method="mc",
        create_eval_env=True,
    )
    model.collect_expert_rollout(nb_rollouts=5)
    model.learn(
        nb_epochs=2,
        nb_rollouts=2,
        log_interval=1,
    )


# def test_debug():
#     import gym
#     from visu.visu_critics import plot_cartpole_critic
#
#     model = REINFORCE(
#         "MlpPolicy",
#         "CartPole-v1",
#         gradient_name="sum",
#         seed=3,
#         verbose=1,
#         critic_estim_method="mc",
#         gamma=0.999,
#         gae_lambda=0.98,
#     )
#
#     model.learn(int(1e5), log_interval=10, nb_rollouts=10)
#
#     plot_cartpole_critic(model.policy, gym.make("CartPole-v1"), figname="test", plot=True)
