import os
from typing import Callable, List, Optional, Tuple
import pklot_env
import supersuit as ss
from datetime import datetime
from os import path as os_path
import torch
import numpy as np
from scipy import interpolate
from model import CNNDQN, Critic

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv, PettingZooEnv
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger

from tianshou.policy import BasePolicy, DiscreteSACPolicy, MultiAgentPolicyManager
from tianshou.trainer import offpolicy_trainer

from tianshou.utils import WandbLogger
from torch.utils.tensorboard import SummaryWriter

cwd = os_path.dirname(__file__)
now = datetime.now()
timestamp = now.strftime("%m-%d-%Y_%H-%M-%S")

MODEL_NAME = "Tianshou-Multiagent"
NUM_AGENT = 4


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
        actor = CNNDQN(
            state_shape=(10, 3, 140, 140),
            action_shape=7,
            obs=env.observation_space.sample()[None],
            device="cuda" if torch.cuda.is_available() else "cpu",
        ).to("cuda" if torch.cuda.is_available() else "cpu")

        critic1 = Critic(
            state_shape=(10, 3, 140, 140),
            obs=env.observation_space.sample()[None],
            device="cuda" if torch.cuda.is_available() else "cpu",
        ).to("cuda" if torch.cuda.is_available() else "cpu")

        critic2 = Critic(
            state_shape=(10, 3, 140, 140),
            obs=env.observation_space.sample()[None],
            device="cuda" if torch.cuda.is_available() else "cpu",
        ).to("cuda" if torch.cuda.is_available() else "cpu")

        optim1 = torch.optim.Adam(actor.parameters(), lr=1e-4)
        optim2 = torch.optim.Adam(critic1.parameters(), lr=1e-4)
        optim3 = torch.optim.Adam(critic2.parameters(), lr=1e-4)

        agents.append(
            DiscreteSACPolicy(
                actor=actor,
                actor_optim=optim1,
                critic1=critic1,
                critic1_optim=optim2,
                critic2=critic2,
                critic2_optim=optim3,
                gamma=0.99,
            )
        )

    policy = MultiAgentPolicyManager(agents, env)
    return policy, optim, env.agents


if __name__ == "__main__":
    # https://pettingzoo.farama.org/tutorials/tianshou/intermediate/
    # ======== Step 1: Environment setup =========
    # TODO: still don't quite get why we need dummy vectors
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
    test_collector = Collector(policy, test_env)
    # policy.set_eps(0.2)
    train_collector.collect(n_episode=20)  # batch size * training_num TODO


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
        pass


    def test_fn(epoch, env_step):
        pass


    def reward_metric(rews):
        return np.average(rews, axis=1)


    # logger:
    # script_path = os.path.dirname(os.path.abspath(__file__))
    # log_path = os.path.join(script_path, f"log/dqn/run{timestamp}")
    # writer = SummaryWriter(log_path)
    # logger = TensorboardLogger(writer)
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
        episode_per_test=10,
        batch_size=256,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        update_per_step=0.25,
        test_in_train=False,
        reward_metric=reward_metric,
        # logger=logger,
    )

    # return result, policy.policies[agents[1]]
    print(f"\n==========Result==========\n{result}")
    print("\n(the trained policy can be accessed via policy.policies[agents[1]])")
