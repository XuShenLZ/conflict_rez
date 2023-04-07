import os
from typing import Callable, List, Optional, Tuple
import pklot_env_unicycle_cont
import supersuit as ss
from datetime import datetime
from os import path as os_path
import torch
import numpy as np
from scipy import interpolate
from model import DuelingDQN, Actor, Critic
import gymnasium as gym
from collections import deque

from tianshou.data import Collector, VectorReplayBuffer, PrioritizedVectorReplayBuffer
from tianshou.env import DummyVectorEnv, PettingZooEnv, SubprocVectorEnv
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger

from tianshou.policy import BasePolicy, DQNPolicy, MultiAgentPolicyManager, DDPGPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils.net.common import Net, ActorCritic

from tianshou.utils import WandbLogger
import wandb
from torch.utils.tensorboard import SummaryWriter
from tianshou_experiment import render_unicycle

cwd = os_path.dirname(__file__)
now = datetime.now()
timestamp = now.strftime("%m-%d-%Y_%H-%M-%S")

MODEL_NAME = "Tianshou-Multiagent"
NUM_AGENT = 1
device = "cuda" if torch.cuda.is_available() else "cpu"

params = pklot_env_unicycle_cont.EnvParams(
    reward_stop=0
)


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


def get_env(render_mode="human"):
    """This function is needed to provide callables for DummyVectorEnv."""
    env = pklot_env_unicycle_cont.raw_env(
        n_vehicles=NUM_AGENT,
        random_reset=False,
        seed=1,
        max_cycles=500,
        render_mode=render_mode,
        params=params
    )  # seed=1
    env = ss.black_death_v3(env)
    env = ss.resize_v1(env, 140, 140)
    return PettingZooEnv(env)


def get_agents() -> Tuple[BasePolicy, List[torch.optim.Optimizer], list]:
    env = get_env()
    agents = []
    optims = []
    for _ in range(NUM_AGENT):
        actor = Actor(
            state_shape=(10, 3, 140, 140),
            action_shape=2,
            obs=env.observation_space.sample()[None],
            device="cuda" if torch.cuda.is_available() else "cpu",
        ).to("cuda" if torch.cuda.is_available() else "cpu")

        critic = Critic(
            state_shape=(10, 3, 140, 140),
            obs=env.observation_space.sample()[None],
            device="cuda" if torch.cuda.is_available() else "cpu",
        ).to("cuda" if torch.cuda.is_available() else "cpu")

        actor_optim = torch.optim.Adam(actor.parameters(), lr=1e-4)
        critic_optim = torch.optim.Adam(critic.parameters(), lr=1e-4)

        agents.append(
            DDPGPolicy(
                actor=actor,
                actor_optim=actor_optim,
                critic=critic,
                critic_optim=critic_optim,
                gamma=0.9,
                estimation_step=NUM_AGENT,
            ).to("cuda" if torch.cuda.is_available() else "cpu")
        )

    policy = MultiAgentPolicyManager(agents, env)
    return policy, optims, env.agents


if __name__ == "__main__":
    # https://pettingzoo.farama.org/tutorials/tianshou/intermediate/
    # ======== Step 1: Environment setup =========
    # TODO: still don't quite get why we need dummy vectors
    train_env = SubprocVectorEnv([get_env for _ in range(1)])
    test_env = DummyVectorEnv([get_env for _ in range(1)])

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
        # PrioritizedVectorReplayBuffer(100000, len(train_env), alpha=0.5, beta=0.4),
        VectorReplayBuffer(100000, len(train_env)),
        exploration_noise=True,
    )
    test_collector = Collector(policy, test_env)
    train_collector.collect(n_episode=100)  # batch size * training_num TODO


    # ======== Step 4: Callback functions setup =========
    # logger:
    # logger = WandbLogger(
    #     project="confrez-tianshou", name=f"ddpg{timestamp}", save_interval=50
    # )
    # script_path = os.path.dirname(os.path.abspath(__file__))
    # log_path = os.path.join(script_path, f"log/dqn/run{timestamp}")
    # writer = SummaryWriter(log_path)
    # logger.load(writer)

    def save_best_fn(policy):
        os.makedirs(os.path.join("log", "dqn"), exist_ok=True)
        for i, agent in enumerate(agents):
            model_save_path = os.path.join("log", "dqn", f"policy{i}.pth")
            torch.save(policy.policies[agent].state_dict(), model_save_path)
            # logger.wandb_run.save(model_save_path)
        render_unicycle(agents, policy, n_vehicles=NUM_AGENT)
        # logger.wandb_run.log({"video": wandb.Video("out.gif", fps=4, format="gif")})


    def stop_over_n(n=10):
        mean_n = deque(maxlen=n)

        def stop_fn(mean_rewards):
            # currently set to never stop
            mean_n.append(mean_rewards)
            return np.mean(mean_n) >= 9950

        return stop_fn



    def train_fn(epoch, env_step):
        # print(env_step, policy.policies[agents[0]]._iter)
        pass


    def test_fn(epoch, env_step):
        for agent in agents:
            policy.policies[agent]


    def reward_metric(rews):
        return np.average(rews, axis=1)


    # ======== Step 5: Run the trainer =========
    result = offpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=10000,
        step_per_epoch=500,
        step_per_collect=10,
        episode_per_test=1,
        batch_size=32 * NUM_AGENT,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_over_n(),
        save_best_fn=save_best_fn,
        update_per_step=0.1,
        test_in_train=False,
        reward_metric=reward_metric,
        # logger=logger,
    )

    # return result, policy.policies[agents[1]]
    print(f"\n==========Result==========\n{result}")
    print("\n(the trained policy can be accessed via policy.policies[agents[1]])")
