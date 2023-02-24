import os
from typing import Callable, List, Optional, Tuple
import pklot_env
import supersuit as ss
from datetime import datetime
from os import path as os_path
import torch
import numpy as np
from scipy import interpolate
from model import CNN_DQN, CNNDQN

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv, PettingZooEnv
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger

from tianshou.policy import (
    BasePolicy,
    DQNPolicy,
    MultiAgentPolicyManager
)
from tianshou.trainer import offpolicy_trainer

from tianshou.utils import WandbLogger
from torch.utils.tensorboard import SummaryWriter

cwd = os_path.dirname(__file__)
now = datetime.now()
timestamp = now.strftime("%m-%d-%Y_%H-%M-%S")

MODEL_NAME = "Tianshou-Multiagent"
NUM_AGENT = 1


def step_schedule(
        initial_value: float, steps: List[float], levels: List[float]
) -> Callable[[float], float]:
    """
    Step learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    weight_func = interpolate.interp1d(steps, levels, "next", fill_value="extrapolate")

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        weight = weight_func(progress_remaining).__float__()
        return weight * initial_value

    return func


def get_env():
    """This function is needed to provide callables for DummyVectorEnv."""
    env = pklot_env.raw_env(n_vehicles=NUM_AGENT, random_reset=False, seed=1)  # seed=1
    env = ss.black_death_v3(env)
    env = ss.resize_v1(env, 140, 140)
    return PettingZooEnv(env)


def get_agents(
        optim: Optional[torch.optim.Optimizer] = None,
) -> Tuple[BasePolicy, torch.optim.Optimizer, list]:
    env = get_env()
    agents = []
    for _ in range(NUM_AGENT):
        net = CNNDQN(
            state_shape=(10, 3, 140, 140),
            action_shape=7,
            obs=env.observation_space.sample()[None],
            device="cuda" if torch.cuda.is_available() else "cpu"
        ).to("cuda" if torch.cuda.is_available() else "cpu")

        if optim is None:
            optim = torch.optim.Adam(net.parameters(), lr=1e-4)

        agents.append(DQNPolicy(
            model=net,
            optim=optim,
            discount_factor=0.99,
            estimation_step=3,
            target_update_freq=5000,
            # reward_normalization=True,
            clip_loss_grad=True,
            lr_scheduler=step_schedule(0.0005, [1, 0.8, 0.6, 0.3], [1, 0.5, 0.1, 0.05])
        ))

    policy = MultiAgentPolicyManager(agents, env)
    return policy, optim, env.agents


if __name__ == "__main__":
    # https://pettingzoo.farama.org/tutorials/tianshou/intermediate/
    # ======== Step 1: Environment setup =========
    # TODO: still don't quite get why we need dummy vectors
    # SubprocVectorEnv: 可以试一下
    # replay buffer 加多少东西
    train_env = DummyVectorEnv([get_env for _ in range(10)])
    test_env = DummyVectorEnv([get_env for _ in range(10)])

    # seed
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ======== Step 2: Agent setup =========
    policy, optim, agents = get_agents()

    # ======== Step 3: Collector setup =========

    train_collector = Collector(
        policy,
        train_env,
        VectorReplayBuffer(100000, len(train_env)),
        exploration_noise=True,
    )
    test_collector = Collector(policy, test_env, exploration_noise=True)
    # policy.set_eps(0.2)
    train_collector.collect(n_episode=5)  # batch size * training_num TODO
    # ======== Step 4: Callback functions setup =========

    def save_best_fn(policy):
        model_save_path = os.path.join("log", "rps", "dqn", "policy.pth")
        os.makedirs(os.path.join("log", "rps", "dqn"), exist_ok=True)
        for agent in agents:
            torch.save(policy.policies[agent].state_dict(), model_save_path)
        # TODO resolved


    def stop_fn(mean_rewards):
        # currently set to never stop
        return False


    def train_fn(epoch, env_step):
        # print(env_step, policy.policies[agents[0]]._iter, env_step)
        for agent in agents:
            policy.policies[agent].set_eps(max(0.97**epoch, 0.1))


    def test_fn(epoch, env_step):
        for agent in agents:
            policy.policies[agent].set_eps(0.0)
        if epoch == 5:
            print(epoch)
            pass

        # temp = policy.policies[agents[0]]
        # temp.eval()
        # temp.set_eps(0.0)
        # collector = Collector(policy, train_env, exploration_noise=True)
        # collector.collect(n_episode=1, render=1 / 35)


    def reward_metric(rews):
        return np.average(rews)


    # logger:
    log_path = os.path.join(f"dqn/run{timestamp}")
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)
    # logger = WandbLogger()
    # logger.load(SummaryWriter("./log/"))

    # ======== Step 5: Run the trainer =========
    result = offpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=1000,
        step_per_epoch=200,
        step_per_collect=50,
        episode_per_test=100,
        batch_size=64,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        update_per_step=1,
        test_in_train=False,
        reward_metric=reward_metric,
        logger=logger
    )

    # return result, policy.policies[agents[1]]
    print(f"\n==========Result==========\n{result}")
    print("\n(the trained policy can be accessed via policy.policies[agents[1]])")
