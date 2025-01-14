import os

import gym
import my_gym  # Necessary to see CartPoleContinuous, though PyCharm does not understand this
from arguments import get_args
from chrono import Chrono
from visu.visu_critics import plot_2d_critic, plot_cartpole_critic, plot_nd_critic, plot_pendulum_critic
from visu.visu_policies import plot_2d_policy, plot_cartpole_policy, plot_nd_policy, plot_pendulum_policy
from visu.visu_trajectories import plot_trajectory

from stable_baselines3 import CEM, REINFORCE
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.reinforce.incomplete_buffer import IncompleteBuffer


def plot_policy(model, env, env_name, gradient_name, final_string="post"):
    actname = env_name + "_actor_" + gradient_name + "_" + final_string + ".pdf"
    if env_name == "Pendulum-v0":
        plot_pendulum_policy(model.policy, env, deterministic=True, figname=actname, plot=False)
    elif env_name == "CartPole-v1" or env_name == "CartPoleContinuous-v0":
        plot_cartpole_policy(model.policy, env, deterministic=True, figname=actname, plot=False)
    elif env.observation_space.shape[0] == 2:
        plot_2d_policy(model.policy, env, deterministic=True, figname=actname, plot=False)
    else:
        plot_nd_policy(model.policy, env, deterministic=True, figname=actname, plot=False)


def plot_critic(model, env, env_name, gradient_name, final_string="post"):
    critname = env_name + "_critic_" + gradient_name + "_" + final_string + ".pdf"
    if env_name == "Pendulum-v0":
        plot_pendulum_critic(model.policy, env, figname=critname, plot=False)
    elif env_name == "CartPole-v1" or env_name == "CartPoleContinuous-v0":
        plot_cartpole_critic(model.policy, env, figname=critname, plot=False)
    elif env.observation_space.shape[0] == 2:
        plot_2d_critic(model.policy, env, figname=critname, plot=False)
    else:
        plot_nd_critic(model.policy, env, figname=critname, plot=False)


def init_test_reinforce():
    args = get_args()
    # env_vec = make_vec_env(args.env_name, n_envs=10, seed=0, vec_env_cls=DummyVecEnv)
    model = REINFORCE(
        "MlpPolicy",
        args.env_name,
        gradient_name="gae",
        seed=1,
        verbose=1,
    )
    model.learn(int(1e5))


def test_reinforce() -> None:
    plot_policies = True
    args = get_args()
    chrono = Chrono()
    # Create log dir
    log_dir = "data/save/"
    os.makedirs(log_dir, exist_ok=True)
    # args.env_name = "Pendulum-v0"
    args.env_name = "CartPole-v1"
    # args.gradients = ["n step","baseline","gae"]
    # args.gradients = ["discount"]
    # args.gradients = ["sum", "discount", "normalized sum", "normalized discount", "gae"]
    args.gradients = ["normalized discount", "gae"]
    args.nb_rollouts = 25
    # When a critic estimation method is specified
    # it is automatically used as a baseline
    args.critic_estim_method = "mc"
    # Create and wrap the environment
    env = gym.make(args.env_name)
    env_vec = make_vec_env(args.env_name, n_envs=10, seed=0, vec_env_cls=DummyVecEnv)
    grads = args.gradients
    for i in range(len(grads)):
        file_name = grads[i] + "_" + args.env_name
        log_file_name = log_dir + file_name
        print(grads[i])
        # lcb = LossCallback(log_dir, file_name)
        eval_callback = EvalCallback(
            env_vec,
            best_model_save_path=log_dir + "bests/",
            log_path=log_dir,
            eval_freq=500,
            n_eval_episodes=5,
            deterministic=True,
            render=False,
        )
        policy_kwargs = dict(net_arch=dict(pi=[100, 100], vf=[100, 100]), optimizer_kwargs=dict(eps=1e-5))

        model = REINFORCE(
            "MlpPolicy",
            env,
            gradient_name=grads[i],
            beta=args.beta,
            gamma=args.gamma,
            learning_rate=args.lr_actor,
            n_steps=args.n_steps,
            # seed=1, #removed to get different rollouts each time
            verbose=1,
            policy_kwargs=policy_kwargs,
            tensorboard_log=log_file_name,
            critic_estim_method=args.critic_estim_method,
            buffer_class=IncompleteBuffer,
        )
        if plot_policies:
            plot_policy(model, env, args.env_name, grads[i], final_string="pre")
            plot_critic(model, env, args.env_name, grads[i], final_string="pre")

        model.learn(
            total_timesteps=50000,
            # nb_epochs=10 * args.nb_repet,
            nb_rollouts=args.nb_rollouts,
            callback=eval_callback,
            log_interval=args.log_interval,
        )

        if plot_policies:
            plot_policy(model, env, args.env_name, grads[i], final_string="post")
            plot_critic(model, env, args.env_name, grads[i], final_string="post")

    chrono.stop()
    # plot_results(args)


def test_imitation_cmc() -> None:
    plot_policies = True
    args = get_args()
    chrono = Chrono()
    # Create log dir
    log_dir = "data/save/"
    os.makedirs(log_dir, exist_ok=True)
    args.env_name = "MountainCarContinuous-v0"
    # args.gradients = ["discount", "normalized sum", "normalized discounted", "sum", "n step", "gae"]
    args.gradients = ["discount"]
    args.nb_rollouts = 8
    args.critic_estim_method = None
    # Create and wrap the environment
    env = gym.make(args.env_name)
    env_vec = make_vec_env(args.env_name, n_envs=10, seed=0, vec_env_cls=DummyVecEnv)
    grads = args.gradients
    for i in range(len(grads)):
        file_name = grads[i] + "_" + args.env_name
        log_file_name = log_dir + file_name
        print(grads[i])
        eval_callback = EvalCallback(
            env_vec,
            best_model_save_path=log_dir + "bests/",
            log_path=log_dir,
            eval_freq=500,
            n_eval_episodes=50,
            deterministic=True,
            render=False,
        )
        policy_kwargs = dict(net_arch=dict(pi=[5, 5], vf=[10, 10]))

        model = REINFORCE(
            "MlpPolicy",
            env,
            gradient_name=grads[i],
            beta=args.beta,
            gamma=args.gamma,
            learning_rate=args.lr_actor,
            n_steps=args.n_steps,
            seed=1,
            verbose=1,
            policy_kwargs=policy_kwargs,
            tensorboard_log=log_file_name,
            critic_estim_method=args.critic_estim_method,
        )
        if plot_policies:
            plot_policy(model, env, args.env_name, grads[i], final_string="pre")

        eval_callback2 = EvalCallback(
            env_vec,
            best_model_save_path=log_dir + "bests/",
            log_path=log_dir,
            eval_freq=500000,
            n_eval_episodes=1,
            deterministic=True,
            render=False,
        )
        model.collect_expert_rollout(nb_rollouts=20, callback=eval_callback2)
        rollout_data = model.rollout_buffer.get_samples()
        plot_trajectory(rollout_data, env, 1)
        plot_policy(model, env, args.env_name, grads[i], final_string="imit")
        args.nb_rollouts = 20
        model.learn(
            nb_epochs=20 * args.nb_repet,
            nb_rollouts=args.nb_rollouts,
            callback=eval_callback,
            log_interval=args.log_interval,
        )

        if plot_policies:
            plot_policy(model, env, args.env_name, grads[i], final_string="post")

    chrono.stop()


def test_cem() -> None:
    plot_policies = False
    args = get_args()
    chrono = Chrono()
    # Create log dir
    log_dir = "data/save/"
    os.makedirs(log_dir, exist_ok=True)
    env_name = "Pendulum-v0"
    args.nb_rollouts = 8
    env = gym.make(env_name)
    file_name = "cem_" + env_name
    log_file_name = log_dir + file_name
    eval_callback = EvalCallback(
        env,
        best_model_save_path=log_dir + "bests/",
        log_path=log_dir,
        eval_freq=1,
        n_eval_episodes=2,
        deterministic=True,
        render=False,
    )
    policy_kwargs = dict(net_arch=[32])

    model = CEM(
        "MlpPolicy",
        env_name,
        seed=1,
        verbose=1,
        noise_multiplier=0.999,
        n_eval_episodes=4,
        sigma=0.2,
        pop_size=20,
        policy_kwargs=policy_kwargs,
        tensorboard_log=log_file_name,
    )
    if plot_policies:
        plot_policy(model, env, env_name, "cem", final_string="pre")

    model.learn(total_timesteps=1e6, callback=eval_callback, log_interval=20)
    if plot_policies:
        plot_policy(model, env, env_name, "cem", final_string="post")

    chrono.stop()


if __name__ == "__main__":
    # init_test_reinforce()
    # test_reinforce()
    # test_imitation_cmc()
    test_cem()
