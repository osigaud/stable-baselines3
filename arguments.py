import argparse

# the following functions are used to build file names for saving data and displaying results


def make_study_string(params):
    return (
        params.env_name
        + "_"
        + params.critic_update_method
        + "_"
        + params.critic_estim_method
        + "_eval_"
        + str(params.deterministic_eval)
    )


def make_study_params_string(params):
    return "cycles_" + str(params.nb_cycles) + "_trajs_" + str(params.nb_rollouts) + "_batches_" + str(params.nb_batches)


def make_learning_params_string(params):
    return (
        "gamma_"
        + str(params.gamma)
        + "_nstep_"
        + str(params.nstep)
        + "_lr_act_"
        + str(params.lr_actor)
        + "_lr_critic_"
        + str(params.lr_critic)
    )


def make_full_string(params):
    return make_study_string(params) + "_" + make_study_params_string(params) + "_" + make_learning_params_string(params)


def get_args():
    """
    Standard function to specify the default value of the hyper-parameters of all policy gradient algorithms
    and experimental setups
    :return: the complete list of arguments
    """
    parser = argparse.ArgumentParser()
    # environment setting
    parser.add_argument("--env_name", type=str, default="CartPoleContinuous-v0", help="the environment name")
    parser.add_argument("--env_obs_space_name", type=str, default=["pos", "angle"])  # ["pos", "angle", "vx", "v angle"]
    parser.add_argument("--render", type=bool, default=False, help="visualize the run or not")
    parser.add_argument("--reward_shift", type=float, default=0.0, help="reward normalization")
    # study settings
    parser.add_argument("--critic_update_method", type=str, default="dataset", help="critic update method: batch or dataset")

    parser.add_argument("--team_name", type=str, default="default_team", help="team name")
    parser.add_argument("--deterministic_eval", type=bool, default=True, help="deterministic policy evaluation?")
    # study parameters
    parser.add_argument("--nb_repet", type=int, default=1, help="number of repetitions to get statistics")
    parser.add_argument("--nb_rollouts", type=int, default=10, help="number of rollouts in a MC batch")
    parser.add_argument("--log_interval", type=int, default=100, help="number of steps between two logs")
    # algo settings
    parser.add_argument(
        "--gradients", type=str, nargs="+", default=["gae", "sum", "discount", "normalize"], help="other: baseline, beta"
    )
    parser.add_argument("--critic_estim_method", type=str, default="td", help="critic estimation method: mc, td or nstep")
    # learning parameters
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--lr_actor", type=float, default=0.01, help="learning rate of the actor")
    parser.add_argument("--lr_critic", type=float, default=0.01, help="learning rate of the critic")
    parser.add_argument("--beta", type=float, default=0.1, help="temperature in AWR-like learning")
    parser.add_argument("--n_steps", type=int, default=5, help="n in n-step return")

    args = parser.parse_args()
    return args
