import os
import random

import matplotlib.pyplot as plt
import numpy as np


def final_show(save_figure, plot, figure_name, x_label, y_label, title, directory) -> None:
    """
    Finalize all plots, adding labels and putting the corresponding file in the specified directory
    :param save_figure: boolean stating whether the figure should be saved
    :param plot: whether the plot should be shown interactively
    :param figure_name: the name of the file where to save the figure
    :param x_label: label on the x axis
    :param y_label: label on the y axis
    :param title: title of the figure
    :param directory: the directory where to save the file
    :return: nothing
    """
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    if save_figure:
        directory = os.getcwd() + "/data" + directory
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(directory + figure_name)
    if plot:
        plt.show()
    plt.close()


def plot_policy(policy, env, deterministic, name, study_name, default_string, num, plot=False) -> None:
    """
    The main entry point for plotting a policy: determine which plotting function to call depending on the
    environment parameters
    :param policy: the policy to be plotted
    :param env: the evaluation environment
    :param deterministic: whether the deterministic version of the policy should be plotted
    :param name: '_ante_' or '_post_' to determine if the policy was plotted before or after training
    :param study_name: the name of the study
    :param default_string: a default string to further specify the plot name
    :param num: a number to save several files corresponding to the same configuration
    :param plot: whether the plot should be interactive
    :return: nothing
    """
    obs_size = env.observation_space.shape[0]
    actor_picture_name = str(num) + "_actor_" + study_name + "_" + default_string + name + ".pdf"
    if obs_size == 1:
        plot_1d_policy(policy, env, deterministic, plot, figname=actor_picture_name)
    elif obs_size == 2:
        plot_2d_policy(policy, env, deterministic, plot, figname=actor_picture_name)
    else:
        plot_nd_policy(policy, env, deterministic, plot, figname=actor_picture_name)


def plot_1d_policy(policy, env, deterministic, plot=True, figname="1D_actor.pdf", save_figure=True) -> None:
    """
    visualization of the policy for a 1D environment like 1D Toy with continuous actions
    :param policy: the policy to be plotted
    :param env: the evaluation environment
    :param deterministic: whether the deterministic version of the policy should be plotted
    :param plot: whether the plot should be interactive
    :param figname: the name of the file to save the figure
    :param save_figure: whether the figure should be saved
    :return: nothing
    """
    if env.observation_space.shape[0] != 1:
        raise (ValueError("The observation space dimension is {}, should be 1".format(env.observation_space.shape[0])))
    definition = 200
    x_min = env.observation_space.low[0]
    x_max = env.observation_space.high[0]

    states = []
    actions = []
    for _, x in enumerate(np.linspace(x_min, x_max, num=definition)):
        obs = np.array([x])
        action, _ = policy.predict(obs, deterministic=deterministic)
        states.append(obs)
        actions.append(action)

    plt.figure(figsize=(10, 10))
    plt.plot(states, actions)
    x_label, y_label = getattr(env.observation_space, "names", ["x", "y"])
    final_show(save_figure, plot, figname, x_label, y_label, "1D Policy", "/plots/")


def plot_2d_policy(policy, env, deterministic, plot=True, figname="2d_actor.pdf", save_figure=True) -> None:
    """
    Plot a policy for a 2D environment like continuous mountain car
    :param policy: the policy to be plotted
    :param env: the evaluation environment
    :param deterministic: whether the deterministic version of the policy should be plotted
    :param plot: whether the plot should be interactive
    :param figname: the name of the file to save the figure
    :param save_figure: whether the figure should be saved
    :return: nothing
    """
    if env.observation_space.shape[0] != 2:
        raise (ValueError("Observation space dimension {}, should be 2".format(env.observation_space.shape[0])))
    definition = 100
    portrait = np.zeros((definition, definition))
    x_min, y_min = env.observation_space.low
    x_max, y_max = env.observation_space.high

    for index_x, x in enumerate(np.linspace(x_min, x_max, num=definition)):
        for index_y, y in enumerate(np.linspace(y_min, y_max, num=definition)):
            # Be careful to fill the matrix in the right order
            obs = np.array([[x, y]])
            action, _ = policy.predict(obs, deterministic=deterministic)
            portrait[definition - (1 + index_y), index_x] = action
    plt.figure(figsize=(10, 10))
    plt.imshow(portrait, cmap="inferno", extent=[x_min, x_max, y_min, y_max], aspect="auto")
    plt.colorbar(label="action")
    # Add a point at the center
    plt.scatter([0], [0])
    x_label, y_label = getattr(env.observation_space, "names", ["x", "y"])
    final_show(save_figure, plot, figname, x_label, y_label, "Actor phase portrait", "/plots/")


def plot_bernoulli_policy(policy, env, plot=True, figure_name="proba_actor.pdf", save_figure=True) -> None:
    """
    Plot the underlying thresholds of a Bernoulli policy for a 2D environment like continuous mountain car.
    :param policy: the policy to be plotted
    :param env: the evaluation environment
    :param plot: whether the plot should be interactive
    :param figure_name: the name of the file to save the figure
    :param save_figure: whether the figure should be saved
    :return: nothing
    """
    if env.observation_space.shape[0] != 2:
        raise (ValueError("Observation space dimension {}, should be 2".format(env.observation_space.shape[0])))
    definition = 100
    portrait = np.zeros((definition, definition))
    x_min, y_min = env.observation_space.low
    x_max, y_max = env.observation_space.high

    for index_x, x in enumerate(np.linspace(x_min, x_max, num=definition)):
        for index_y, y in enumerate(np.linspace(y_min, y_max, num=definition)):
            # Be careful to fill the matrix in the right order
            state = np.array([[x, y]])
            probs = policy.forward(state)
            action = probs.data.numpy().astype(float)
            portrait[definition - (1 + index_y), index_x] = action
    plt.figure(figsize=(10, 10))
    plt.imshow(portrait, cmap="inferno", extent=[x_min, x_max, y_min, y_max], aspect="auto")
    plt.colorbar(label="action")
    # Add a point at the center
    plt.scatter([0], [0])
    x_label, y_label = getattr(env.observation_space, "names", ["x", "y"])
    final_show(save_figure, plot, figure_name, x_label, y_label, "Actor phase portrait", "/plots/")


def plot_nd_policy(policy, env, deterministic, plot=True, figname="nd_actor.pdf", save_figure=True) -> None:
    """
    Plot a policy for a ND environment
    :param policy: the policy to be plotted
    :param env: the evaluation environment
    :param deterministic: whether the deterministic version of the policy should be plotted
    :param plot: whether the plot should be interactive
    :param figname: the name of the file to save the figure
    :param save_figure: whether the figure should be saved
    :return: nothing
    """
    if env.observation_space.shape[0] <= 2:
        raise (ValueError("Observation space dimension {}, should be > 2".format(env.observation_space.shape[0])))
    definition = 100
    portrait = np.zeros((definition, definition))
    state_min = env.observation_space.low
    state_max = env.observation_space.high
    # Use the dimension names if given otherwise default to "x" and "y"

    for index_x, x in enumerate(np.linspace(state_min[0], state_max[0], num=definition)):
        for index_y, y in enumerate(np.linspace(state_min[1], state_max[1], num=definition)):
            obs = np.array([[x, y]])
            for _ in range(2, len(state_min)):
                z = random.random() - 0.5
                obs = np.append(obs, z)
            action, _ = policy.predict(obs, deterministic=deterministic)
            portrait[definition - (1 + index_y), index_x] = action
    plt.figure(figsize=(10, 10))
    plt.imshow(portrait, cmap="inferno", extent=[state_min[0], state_max[0], state_min[1], state_max[1]], aspect="auto")
    plt.colorbar(label="action")
    # Add a point at the center
    plt.scatter([0], [0])
    x_label, y_label = getattr(env.observation_space, "names", ["x", "y"])
    final_show(save_figure, plot, figname, x_label, y_label, "Actor phase portrait", "/plots/")


def plot_cartpole_policy(policy, env, deterministic, plot=True, figname="cartpole_actor.pdf", save_figure=True) -> None:
    """
    Plot a policy for a cartpole environment
    :param policy: the policy to be plotted
    :param env: the evaluation environment
    :param deterministic: whether the deterministic version of the policy should be plotted
    :param plot: whether the plot should be interactive
    :param figname: the name of the file to save the figure
    :param save_figure: whether the figure should be saved
    :return: nothing
    """
    if env.observation_space.shape[0] <= 2:
        raise (ValueError("Observation space dimension {}, should be > 2".format(env.observation_space.shape[0])))
    definition = 100
    portrait = np.zeros((definition, definition))
    state_min = env.observation_space.low
    state_max = env.observation_space.high

    for index_x, x in enumerate(np.linspace(state_min[0], state_max[0], num=definition)):
        for index_y, y in enumerate(np.linspace(state_min[2], state_max[2], num=definition)):
            obs = np.array([x])
            z1 = random.random() - 0.5
            z2 = random.random() - 0.5
            obs = np.append(obs, z1)
            obs = np.append(obs, y)
            obs = np.append(obs, z2)
            action, _ = policy.predict(obs, deterministic=deterministic)
            portrait[definition - (1 + index_y), index_x] = action
    plt.figure(figsize=(10, 10))
    plt.imshow(portrait, cmap="inferno", extent=[state_min[0], state_max[0], state_min[2], state_max[2]], aspect="auto")
    plt.colorbar(label="action")
    # Add a point at the center
    plt.scatter([0], [0])
    x_label, y_label = getattr(env.observation_space, "names", ["x", "y"])
    final_show(save_figure, plot, figname, x_label, y_label, figname, "/plots/")


def plot_pendulum_policy(policy, env, deterministic, plot=True, figname="pendulum_actor.pdf", save_figure=True) -> None:
    """
    Plot a policy for the Pendulum environment
    :param policy: the policy to be plotted
    :param env: the evaluation environment
    :param deterministic: whether the deterministic version of the policy should be plotted
    :param plot: whether the plot should be interactive
    :param figname: the name of the file to save the figure
    :param save_figure: whether the figure should be saved
    :return: nothing
    """
    if env.observation_space.shape[0] <= 2:
        raise (ValueError("Observation space dimension {}, should be > 2".format(env.observation_space.shape[0])))
    definition = 100
    portrait = np.zeros((definition, definition))
    state_min = env.observation_space.low
    state_max = env.observation_space.high

    for index_t, t in enumerate(np.linspace(-np.pi, np.pi, num=definition)):
        for index_td, td in enumerate(np.linspace(state_min[2], state_max[2], num=definition)):
            obs = np.array([[np.cos(t), np.sin(t), td]])
            action, _ = policy.predict(obs, deterministic=deterministic)
            portrait[definition - (1 + index_td), index_t] = action
    plt.figure(figsize=(10, 10))
    plt.imshow(portrait, cmap="inferno", extent=[-180, 180, state_min[2], state_max[2]], aspect="auto")
    plt.colorbar(label="action")
    # Add a point at the center
    plt.scatter([0], [0])
    x_label, y_label = getattr(env.observation_space, "names", ["x", "y"])
    final_show(save_figure, plot, figname, x_label, y_label, figname, "/plots/")
