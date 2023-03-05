import os
from typing import Callable, List, Optional, Tuple
import pklot_env
import supersuit as ss
from datetime import datetime
from os import path as os_path
import torch
import numpy as np
from scipy import interpolate
from model import Actor, Critic

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv, PettingZooEnv
from tianshou.utils.net.common import ActorCritic

from tianshou.policy import BasePolicy, PPOPolicy, MultiAgentPolicyManager
from tianshou.trainer import onpolicy_trainer

from tianshou.utils import WandbLogger, TensorboardLogger
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical

cwd = os_path.dirname(__file__)
now = datetime.now()
timestamp = now.strftime("%m-%d-%Y_%H-%M-%S")

MODEL_NAME = "Tianshou-Multiagent-PPO"
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
    env = pklot_env.raw_env(n_vehicles=NUM_AGENT, random_reset=False, seed=1)
    env = ss.black_death_v3(env)
    env = ss.resize_v1(env, 140, 140)
    return PettingZooEnv(env)


def get_agents(
        optim: Optional[torch.optim.Optimizer] = None,
) -> Tuple[BasePolicy, torch.optim.Optimizer, list]:
    env = get_env()
    agents = []
    for _ in range(NUM_AGENT):
        actor = Actor(
            state_shape=(10, 3, 140, 140),
            action_shape=7,
            obs=env.observation_space.sample()[None],
            device="cuda" if torch.cuda.is_available() else "cpu",
        ).to("cuda" if torch.cuda.is_available() else "cpu")

        critic = Critic(
            state_shape=(10, 3, 140, 140),
            obs=env.observation_space.sample()[None],
            device="cuda" if torch.cuda.is_available() else "cpu",
        ).to("cuda" if torch.cuda.is_available() else "cpu")

        actor_critic = ActorCritic(actor, critic)

        if optim is None:
            optim = torch.optim.Adam(actor_critic.parameters(), lr=1e-4)

        def dist(p):
            return Categorical(logits=p)

        agents.append(
            PPOPolicy(
                actor=actor,
                critic=critic,
                optim=optim,
                discount_factor=0.99,
                dist_fn=Categorical,
                ent_coef=0.01,
                max_grad_norm=0.5
            ) #.to("cuda" if torch.cuda.is_available() else "cpu")
        )

    policy = MultiAgentPolicyManager(agents, env)
    return policy, optim, env.agents


if __name__ == "__main__":
    # https://pettingzoo.farama.org/tutorials/tianshou/intermediate/
    # ======== Step 1: Environment setup =========
    # TODO: still don't quite get why we need dummy vectors
    train_env = DummyVectorEnv([get_env for _ in range(40)])
    test_env = DummyVectorEnv([get_env for _ in range(20)])

    # seed
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ======== Step 2: Agent setup =========
    policy, optim, agents = get_agents()

    # ======== Step 3: Collector setup =========
    buffer = VectorReplayBuffer(
        10000,
        buffer_num=len(train_env),
        ignore_obs_next=True,
        # save_only_last_obs=True,
        # stack_num=4,
    )
    train_collector = Collector(
        policy,
        train_env,
        buffer,
        exploration_noise=True
    )
    test_collector = Collector(policy, test_env)

    # ======== Step 4: Callback functions setup =========

    def save_best_fn(policy):
        model_save_path = os.path.join("log", "ppo", "policy.pth")
        os.makedirs(os.path.join("log", "ppo"), exist_ok=True)
        for agent in agents:
            torch.save(policy.policies[agent].state_dict(), model_save_path)
            logger.wandb_run.save(model_save_path)


    def stop_fn(mean_rewards):
        # currently set to never stop
        return mean_rewards >= 9500


    def train_fn(epoch, env_step):
        return


    def test_fn(epoch, env_step):
        return


    def reward_metric(rews):
        return np.average(rews, axis=1)


    def watch():
        print("Setup test envs ...")
        policy.eval()
        policy.set_eps(0)
        print("Testing agent ...")
        test_collector.reset()
        result = test_collector.collect(
            n_episode=1, render=1/30
        )
        rew = result["rews"].mean()
        print(f"Mean reward (over {result['n/ep']} episodes): {rew}")

    logger = WandbLogger(project="confrez-tianshou", name=f"ppo{timestamp}", save_interval=50)
    script_path = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(script_path, f"log/ppo/run{timestamp}")
    writer = SummaryWriter(log_path)
    # logger = TensorboardLogger(writer)
    logger.load(writer)

    # ======== Step 5: Run the trainer =========
    result = onpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=1000,
        step_per_epoch=200,
        step_per_collect=4000,
        episode_per_test=20,
        batch_size=64,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        repeat_per_collect=40,
        test_in_train=False,
        reward_metric=reward_metric,
        logger=logger,
    )

    # return result, policy.policies[agents[1]]
    print(f"\n==========Result==========\n{result}")
    print("\n(the trained policy can be accessed via policy.policies[agents[1]])")
