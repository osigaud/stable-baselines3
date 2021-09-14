import matplotlib.pyplot as plt
from visu.visu_policies import final_show


def episode_to_traj(rollout_data):
    """
    Transform the states of a rollout_data into a set of (x,y) pairs
    :param rollout_data:
    :return: the (x,y) pairs
    """
    x = []
    y = []
    obs = rollout_data.observations
    # TODO : treat the case where the variables to plot are not the first two
    for o in obs:
        x.append(o[0].numpy())
        y.append(o[1].numpy())
    return x, y


def plot_trajectory(rollout_data, env, fig_index, save_figure=True, plot=True) -> None:
    """
    Plot the set of trajectories stored into a batch
    :param rollout_data: the source batch
    :param env: the environment where the batch was built
    :param fig_index: a number, to save several similar plots
    :param save_figure: where the plot should be saved
    :return: nothing
    """
    if env.observation_space.shape[0] < 2:
        raise (ValueError("Observation space of dimension {}, should be at least 2".format(env.observation_space.shape[0])))

    # Use the dimension names if given otherwise default to "x" and "y"
    x_label, y_label = getattr(env.observation_space, "names", ["x", "y"])

    x, y = episode_to_traj(rollout_data)
    plt.scatter(x, y, c=range(1, len(rollout_data.observations) + 1), s=3)
    figname = "trajectory_" + str(fig_index) + ".pdf"
    final_show(save_figure, plot, figname, x_label, y_label, "Trajectory", "/plots/")
