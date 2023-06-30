import os
from collections import deque
from typing import Callable, List, Optional, Tuple, Union, Dict, Any

import tianshou.data
from tianshou.utils.net.continuous import ActorProb, Critic, Actor

import pklot_env_unicycle_cont as pklot_env
import supersuit as ss
from datetime import datetime
from os import path as os_path
import torch
import numpy as np
from scipy import interpolate

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv, PettingZooEnv, SubprocVectorEnv
from torch.distributions.transforms import TanhTransform

from tianshou.policy import BasePolicy, PPOPolicy, MultiAgentPolicyManager, SACPolicy, DDPGPolicy
from tianshou.trainer import onpolicy_trainer, offpolicy_trainer

from tianshou.utils import WandbLogger, TensorboardLogger
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical, Independent, Normal
from tianshou.exploration import GaussianNoise
from torch import nn
from tianshou_experiment import render_actor_prob, render_actor
import wandb
from model import CriticCont

params = pklot_env.EnvParams(
    reward_stop=-10, reward_dist=-1
)


class CNNDQN(nn.Module):
    def __init__(self, state_shape, device) -> None:
        super(CNNDQN, self).__init__()
        self.device = device
        self.net = nn.Sequential(
            nn.Conv2d(state_shape[1], 32, kernel_size=8, stride=4), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True),
            nn.Flatten()
        )
        with torch.no_grad():
            dim = np.prod(self.net(torch.zeros(state_shape)).shape[1:])

        self.linear = nn.Sequential(nn.Linear(dim, 512), nn.ReLU(inplace=True))
        self.output_dim = 512

    def forward(self, obs, state=None, info={}):
        if type(obs) is tianshou.data.Batch:
            obs = obs['obs']
        obs = obs.reshape(-1, 140, 140, 3)
        if type(obs) is np.ndarray:
            obs = torch.tensor(obs, dtype=torch.float)
        else:
            pass
        obs = torch.transpose(obs, 1, 3)
        # print("DEBUG: obs shape", obs.shape)
        obs = obs.to(device=self.device)
        logits = self.linear(self.net(obs))
        return logits, state


cwd = os_path.dirname(__file__)
now = datetime.now()
timestamp = now.strftime("%m-%d-%Y_%H-%M-%S")
device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "Tianshou-Multiagent-SAC"
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


def get_env(render_mode='human'):
    """This function is needed to provide callables for DummyVectorEnv."""
    env = pklot_env.raw_env(render_mode=render_mode, n_vehicles=NUM_AGENT, random_reset=False, seed=1, params=params)
    env = ss.black_death_v3(env)
    env = ss.resize_v1(env, 140, 140)
    return PettingZooEnv(env)


def get_agents(
        optim: Optional[torch.optim.Optimizer] = None,
) -> Tuple[BasePolicy, torch.optim.Optimizer, list]:
    env = get_env()
    agents = []
    observation_space = (10, 3, 140, 140)
    for _ in range(NUM_AGENT):
        max_action = env.action_space.high[0]
        exploration_noise = 0.1 * max_action
        net = CNNDQN(observation_space, device=device).to(device)
        actor = ActorProb(
            net, action_shape=env.action_space.shape, device=device, max_action=max_action,
            unbounded=True, conditioned_sigma=True
        ).to(device)

        critic1 = CriticCont(observation_space, env.action_space.shape[0], device=device).to(device)
        critic2 = CriticCont(observation_space, env.action_space.shape[0], device=device).to(device)

        actor_optim = torch.optim.Adam(actor.parameters(), lr=1e-4)
        critic1_optim = torch.optim.Adam(critic1.parameters(), lr=1e-5)
        critic2_optim = torch.optim.Adam(critic2.parameters(), lr=1e-5)

        agents.append(
            SACPolicy(
                actor,
                actor_optim,
                critic1,
                critic1_optim,
                critic2,
                critic2_optim,
                gamma=0.99,
                tau=0.005,
                alpha=0.2,
                estimation_step=NUM_AGENT,
                action_space=env.action_space,
                exploration_noise=GaussianNoise(),
            )
        )


    policy = MultiAgentPolicyManager(agents, env)
    return policy, optim, env.agents


if __name__ == "__main__":
    # https://pettingzoo.farama.org/tutorials/tianshou/intermediate/
    # ======== Step 1: Environment setup =========
    # TODO: still don't quite get why we need dummy vectors
    train_env = SubprocVectorEnv([get_env for _ in range(1)])
    test_env = SubprocVectorEnv([get_env for _ in range(1)])

    # seed
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ======== Step 2: Agent setup =========
    policy, optim, agents = get_agents()

    # ======== Step 3: Collector setup =========
    buffer = VectorReplayBuffer(
        100000,
        buffer_num=len(train_env),
    )

    train_collector = Collector(
        policy,
        train_env,
        buffer,
        exploration_noise=True
    )
    train_collector.collect(n_step=10000, random=True)
    test_collector = Collector(policy, test_env)


    # ======== Step 4: Callback functions setup =========

    def save_best_fn(policy):
        model_save_path = os.path.join("log", "ppo", "policy.pth")
        os.makedirs(os.path.join("log", "ppo"), exist_ok=True)
        for agent in agents:
            torch.save(policy.policies[agent].state_dict(), model_save_path)
            # logger.wandb_run.save(model_save_path)
        render_actor_prob(agents, policy, get_env(render_mode='rgb_array'))
        # logger.wandb_run.log({"video": wandb.Video("out.gif", fps=4, format="gif")})


    def stop_over_n(n=10):
        mean_n = deque(maxlen=n)

        def stop_fn(mean_rewards):
            # currently set to never stop
            mean_n.append(mean_rewards)
            return np.mean(mean_n) >= 900


    def train_fn(epoch, env_step):
        return


    def test_fn(epoch, env_step):
        return
        # for agent in agents:
        #     # policy.policies[agent].set_eps(0.1)
        #     policy.policies[agent].set_eps(max(0.997 ** epoch, 0.1))


    def reward_metric(rews):
        return np.average(rews, axis=1)


    def watch():
        print("Setup test envs ...")
        policy.eval()
        policy.set_eps(0)
        print("Testing agent ...")
        test_collector.reset()
        result = test_collector.collect(
            n_episode=1, render=1 / 30
        )
        rew = result["rews"].mean()
        print(f"Mean reward (over {result['n/ep']} episodes): {rew}")


    # logger = WandbLogger(project="confrez-tianshou", name=f"ppo_cont{timestamp}", save_interval=4)
    # script_path = os.path.dirname(os.path.abspath(__file__))
    # log_path = os.path.join(script_path, f"log/ppo/run{timestamp}")
    # writer = SummaryWriter(log_path)
    # logger.load(writer)

    # ======== Step 5: Run the trainer =========
    result = offpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=2000,
        step_per_epoch=500,
        episode_per_test=1,
        step_per_collect=32,
        update_per_step=0.25,
        batch_size=64,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_over_n(5),
        save_best_fn=save_best_fn,
        test_in_train=False,
        reward_metric=reward_metric,
        # logger=logger,
    )

    # return result, policy.policies[agents[1]]
    print(f"\n==========Result==========\n{result}")
    print("\n(the trained policy can be accessed via policy.policies[agents[1]])")
