#!/usr/bin/env python3

import argparse
import datetime
import os
import pprint

import gymnasium
import numpy as np
import torch

from tianshou.env import DummyVectorEnv
from webots_env import make_webots_env

from examples.common import logger_factory
from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer, AsyncCollector
from tianshou.policy import SACPolicy
from tianshou.policy.base import BasePolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
from robot_supervisor_controller import RobotSupervisorController, WebotsLauncher
from vehicle import Car
from gymnasium.wrappers import TimeLimit
from typing import Any


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="VektorE-v0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--buffer-size", type=int, default=1000000)  # 1000000
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[256, 256])
    parser.add_argument("--actor-lr", type=float, default=1e-3)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--auto-alpha", default=True, action="store_true")
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument("--start-timesteps", type=int, default=10000)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--step-per-epoch", type=int, default=5000)
    parser.add_argument("--step-per-collect", type=int, default=1)
    parser.add_argument("--update-per-step", type=int, default=1)
    parser.add_argument("--n-step", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=500)  # 256
    parser.add_argument("--training-num", type=int, default=1)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="log.")
    parser.add_argument("--render", type=float, default=0.0)
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--resume-path", type=str, default="log./VektorE-v0/sac/0/240317-204900/policy.pth")
    parser.add_argument("--resume-id", type=str, default=None)
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
    )
    parser.add_argument("--wandb-project", type=str, default="mujoco.benchmark")
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="watch the play of pre-trained policy only",
    )
    return parser.parse_args()


def test_sac(args: argparse.Namespace = get_args()) -> None:
    # WebotsLauncher().launch_webots(
    #    "/home/elouarn/WebotsProjects/Simulateur_CoVAPSy_Webots2023b_Base2/worlds/Piste_CoVAPSy_2023b.wbt")

    # car = Car()
    # kwargs = dict()
    # kwargs["car"] = car
    # env = gymnasium.make(args.task, **kwargs)
    env, train_envs, test_envs = make_webots_env(
        args.task,
        args.seed,
        args.training_num,
        args.test_num,
        obs_norm=False,
    )
    env = TimeLimit(env, max_episode_steps=1000)
    # env = FlattenObservation(env)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    print("Action range:", np.min(env.action_space.low), np.max(env.action_space.high))
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # model
    net_a = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
    actor = ActorProb(
        net_a,
        args.action_shape,
        device=args.device,
        unbounded=True,
        conditioned_sigma=True,
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    net_c1 = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device,
    )
    net_c2 = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device,
    )
    critic1 = Critic(net_c1, device=args.device).to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2 = Critic(net_c2, device=args.device).to(args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    if args.auto_alpha:
        target_entropy = -np.prod(env.action_space.shape)
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        args.alpha = (target_entropy, log_alpha, alpha_optim)

    policy: SACPolicy = SACPolicy(
        actor=actor,
        actor_optim=actor_optim,
        critic=critic1,
        critic_optim=critic1_optim,
        critic2=critic2,
        critic2_optim=critic2_optim,
        tau=args.tau,
        gamma=args.gamma,
        alpha=args.alpha,
        estimation_step=args.n_step,
        action_space=env.action_space,
    )

    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)

    # collector
    buffer: VectorReplayBuffer | ReplayBuffer
    if args.training_num > 1:
        buffer = VectorReplayBuffer(args.buffer_size, len(train_envs))
    else:
        buffer = ReplayBuffer(args.buffer_size)
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs)
    train_collector.collect(n_step=args.start_timesteps, random=True)

    # log
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    args.algo_name = "sac"
    log_name = os.path.join(args.task, args.algo_name, str(args.seed), now)
    log_path = os.path.join(args.logdir, log_name)

    # logger
    if args.logger == "wandb":
        logger_factory.logger_type = "wandb"
        logger_factory.wandb_project = args.wandb_project
    else:
        logger_factory.logger_type = "tensorboard"

    logger = logger_factory.create_logger(
        log_dir=log_path,
        experiment_name=log_name,
        run_id=args.resume_id,
        config_dict=vars(args),
    )

    def save_best_fn(policy: BasePolicy) -> None:
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    if not args.watch:
        # trainer
        result = OffpolicyTrainer(
            policy=policy,
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=args.epoch,
            step_per_epoch=args.step_per_epoch,
            step_per_collect=args.step_per_collect,
            episode_per_test=args.test_num,
            batch_size=args.batch_size,
            save_best_fn=save_best_fn,
            logger=logger,
            update_per_step=args.update_per_step,
            test_in_train=False,
        ).run()
        pprint.pprint(result)

    # Let's watch its performance!
    policy.eval()
    test_envs.seed(args.seed)
    test_collector.reset()
    collector_stats = test_collector.collect(n_episode=args.test_num, render=args.render)
    print(collector_stats)

    # # load a previous policy
    # if args.mode == "test" and args.resume_path:
    #     policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
    #     print("Loaded agent from: ", args.resume_path)

    # # collector
    # buffer = ReplayBuffer(args.buffer_size)
    # collector = AsyncCollector(policy, env, buffer, exploration_noise=True if args.mode == "train" else False)
    #
    # # log
    # now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    # args.algo_name = "sac"
    # log_name = os.path.join(args.task, args.algo_name, str(args.seed), now)
    # log_path = os.path.join(args.logdir, log_name)
    #
    # # logger
    # if args.logger == "wandb":
    #     logger_factory.logger_type = "wandb"
    #     logger_factory.wandb_project = args.wandb_project
    # else:
    #     logger_factory.logger_type = "tensorboard"
    #
    # logger = logger_factory.create_logger(
    #     log_dir=log_path,
    #     experiment_name=log_name,
    #     run_id=args.resume_id,
    #     config_dict=vars(args),
    # )
    #
    # def save_best_fn(policy: BasePolicy) -> None:
    #     torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))
    #
    # if args.mode == "train":
    #     while car.step() != -1:
    #         print("dans step")
    #         # Start training
    #         collector.collect(n_step=500)
    #         result = OffpolicyTrainer(
    #             policy=policy,
    #             train_collector=collector,
    #             test_collector=None,
    #             max_epoch=args.epoch,
    #             step_per_epoch=args.step_per_epoch,
    #             step_per_collect=args.step_per_collect,
    #             episode_per_test=args.test_num,
    #             batch_size=args.batch_size,
    #             save_best_fn=save_best_fn,
    #             logger=logger,
    #             update_per_step=args.update_per_step,
    #             test_in_train=False,
    #         ).run()
    #         pprint.pprint(result)
    # elif args.mode == "test":
    #     while car.step() != -1:
    #         # Let's watch its performance!
    #         policy.eval()
    #         env.seed(args.seed)
    #         collector.reset()
    #         collector_stats = collector.collect(n_episode=args.test_num, render=args.render)
    #         print(collector_stats)
