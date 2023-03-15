import argparse
import os
import pklot_env
import supersuit as ss
import warnings
from typing import List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pettingzoo.butterfly.pistonball_v6 as pistonball_v6
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import BasePolicy, DQNPolicy, MultiAgentPolicyManager
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net

import os
from typing import Callable, List, Optional, Tuple
import pklot_env
import supersuit as ss
from datetime import datetime
from os import path as os_path
import torch
import numpy as np
from scipy import interpolate
from model import CNNDQN, DuelingDQN

from tianshou.data import Collector, VectorReplayBuffer, PrioritizedVectorReplayBuffer
from tianshou.env import DummyVectorEnv, PettingZooEnv, SubprocVectorEnv
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger

from tianshou.policy import BasePolicy, DQNPolicy, MultiAgentPolicyManager
from tianshou.trainer import offpolicy_trainer

from tianshou.utils import WandbLogger
from torch.utils.tensorboard import SummaryWriter


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1626)
    parser.add_argument('--eps-test', type=float, default=0.0)
    parser.add_argument('--eps-train', type=float, default=0.1)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument(
        '--gamma', type=float, default=0.9, help='a smaller gamma favors earlier win'
    )
    parser.add_argument(
        '--n-pistons',
        type=int,
        default=4,
        help='Number of pistons(agents) in the env'
    )
    parser.add_argument('--n-step', type=int, default=4)
    parser.add_argument('--target-update-freq', type=int, default=5000)
    parser.add_argument('--epoch', type=int, default=10000)
    parser.add_argument('--step-per-epoch', type=int, default=500)
    parser.add_argument('--step-per-collect', type=int, default=10)
    parser.add_argument('--update-per-step', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[64, 64])
    parser.add_argument('--training-num', type=int, default=10)
    parser.add_argument('--test-num', type=int, default=1)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=1/30)

    parser.add_argument(
        '--watch',
        default=True,
        action='store_true',
        help='no training, '
        'watch the play of pre-trained models'
    )
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    return parser


def get_args() -> argparse.Namespace:
    parser = get_parser()
    return parser.parse_known_args()[0]


def get_env(args=get_args()):
    # return PettingZooEnv(pistonball_v6.env(continuous=False, n_pistons=args.n_pistons))
    env = pklot_env.raw_env(n_vehicles=args.n_pistons, random_reset=False, seed=1)
    env = ss.black_death_v3(env)
    env = ss.resize_v1(env, 140, 140)
    return PettingZooEnv(env)


def get_agents(
        args=get_args(),
        agents: Optional[List[BasePolicy]] = None,
        optims: Optional[List[torch.optim.Optimizer]] = None,
        ) -> Tuple[BasePolicy, List[torch.optim.Optimizer], List]:
    env = get_env()
    agents = []
    optims = []
    for _ in range(args.n_pistons):
        net = DuelingDQN(
            state_shape=(10, 3, 140, 140),
            action_shape=7,
            obs=env.observation_space.sample()[None],
            device="cuda" if torch.cuda.is_available() else "cpu",
        ).to("cuda" if torch.cuda.is_available() else "cpu")

        optim = torch.optim.Adam(net.parameters(), lr=1e-4)  # , eps=1.5e-4
        optims.append(optim)
        agents.append(
            DQNPolicy(
                model=net,
                optim=optim,
                discount_factor=0.99,
                estimation_step=4,
                target_update_freq=int(
                    5000
                ),  # update_per_step * number of env steps before updating target
                # clip_loss_grad=True,
            ).to("cuda" if torch.cuda.is_available() else "cpu")
        )

    policy = MultiAgentPolicyManager(agents, env)
    return policy, optims, env.agents


# def get_agents(
#         args=get_args(),
#         agents: Optional[List[BasePolicy]] = None,
#         optims: Optional[List[torch.optim.Optimizer]] = None,
# ) -> Tuple[BasePolicy, List[torch.optim.Optimizer], List]:
#     env = get_env()
#     observation_space = env.observation_space['observation'] if isinstance(
#         env.observation_space, gym.spaces.Dict
#     ) else env.observation_space
#     args.state_shape = observation_space.shape or observation_space.n
#     args.action_shape = env.action_space.shape or env.action_space.n
#     if agents is None:
#         agents = []
#         optims = []
#         for _ in range(args.n_pistons):
#             # model
#             net = Net(
#                 args.state_shape,
#                 args.action_shape,
#                 hidden_sizes=args.hidden_sizes,
#                 device=args.device
#             ).to(args.device)
#             optim = torch.optim.Adam(net.parameters(), lr=args.lr)
#             agent = DQNPolicy(
#                 net,
#                 optim,
#                 args.gamma,
#                 args.n_step,
#                 target_update_freq=args.target_update_freq
#             )
#             agents.append(agent)
#             optims.append(optim)
#
#     policy = MultiAgentPolicyManager(agents, env)
#     return policy, optims, env.agents


def train_agent(
        args=get_args(),
        agents: Optional[List[BasePolicy]] = None,
        optims: Optional[List[torch.optim.Optimizer]] = None,
) -> Tuple[dict, BasePolicy]:
    train_envs = SubprocVectorEnv([get_env for _ in range(args.training_num)])
    test_envs = DummyVectorEnv([get_env for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    policy, optim, agents = get_agents(args, agents=agents, optims=optims)

    # collector
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        exploration_noise=True
    )
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    train_collector.collect(n_step=args.batch_size * args.training_num)

    logger = WandbLogger(project="confrez-tianshou", name=f"rainbow_pistonballcode", save_interval=10)
    script_path = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(script_path, f"log/dqn/pistonballcode")
    writer = SummaryWriter(log_path)
    logger.load(writer)

    def save_best_fn(policy):
        os.makedirs(os.path.join("log", "dqn"), exist_ok=True)
        for i, agent in enumerate(agents):
            model_save_path = os.path.join("log", "dqn", f"policy{i}.pth")
            torch.save(policy.policies[agent].state_dict(), model_save_path)
            logger.wandb_run.save(model_save_path)

    def stop_fn(mean_rewards):
        return mean_rewards > 9000

    def train_fn(epoch, env_step):
        [agent.set_eps(max(0.97 ** epoch, 0.1)) for agent in policy.policies.values()]

    def test_fn(epoch, env_step):
        [agent.set_eps(args.eps_test) for agent in policy.policies.values()]

    def reward_metric(rews):
        return np.average(rews, axis=1)

    # trainer
    result = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.step_per_collect,
        args.test_num,
        args.batch_size,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        update_per_step=args.update_per_step,
        logger=logger,
        test_in_train=False,
        reward_metric=reward_metric
    )

    return result, policy


def watch(
        args=get_args(), policy: Optional[BasePolicy] = None
) -> None:
    env = DummyVectorEnv([get_env])
    if not policy:
        warnings.warn(
            "watching random agents, as loading pre-trained policies is "
            "currently not supported"
        )
        policy, _, _ = get_agents(args)
    policy.eval()
    [agent.set_eps(args.eps_test) for agent in policy.policies.values()]
    collector = Collector(policy, env, exploration_noise=True)
    result = collector.collect(n_episode=1, render=args.render)
    rews, lens = result["rews"], result["lens"]
    print(f"Final reward: {rews[:, 0].mean()}, length: {lens.mean()}")


if __name__ == '__main__':
    result, policy = train_agent()
    watch(policy=policy)
