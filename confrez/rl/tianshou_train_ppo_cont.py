import os
from collections import deque
from typing import Callable, List, Optional, Tuple, Union, Dict, Any

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

from tianshou.policy import BasePolicy, PPOPolicy, MultiAgentPolicyManager
from tianshou.trainer import onpolicy_trainer

from tianshou.utils import WandbLogger, TensorboardLogger
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical, Independent, Normal
from torch import nn
from tianshou_experiment import render_ppo
import wandb

class CNNDQN(nn.Module):
    def __init__(self, state_shape, device, feature_dim=512) -> None:
        super().__init__()
        self.device = device
        self.net = nn.Sequential(
            nn.Conv2d(state_shape[1], 32, kernel_size=8, stride=4), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True),
            nn.Flatten()
        )
        with torch.no_grad():
            self.output_dim = np.prod(self.net(torch.zeros(state_shape)).shape[1:])

    def forward(self, obs, state=None, info={}):
        obs = obs.reshape(-1, 140, 140, 3)
        if type(obs) is np.ndarray:
            obs = torch.tensor(obs, dtype=torch.float)
        obs = torch.transpose(obs, 1, 3)
        # print("DEBUG: obs shape", obs.shape)
        obs = obs.to(device=self.device)

        return self.net(obs), state


class DQN(nn.Module):
    """Reference: Human-level control through deep reinforcement learning.
    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
            self,
            c: int,
            h: int,
            w: int,
            device: Union[str, int, torch.device] = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.c = c
        self.h = h
        self.w = w
        self.net = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True),
            nn.Flatten()
        )
        with torch.no_grad():
            self.output_dim = np.prod(self.net(torch.zeros(1, c, h, w)).shape[1:])

    def forward(
            self,
            x: Union[np.ndarray, torch.Tensor],
            state: Optional[Any] = None,
            info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: x -> Q(x, \*)."""
        x = torch.as_tensor(x, device=self.device, dtype=torch.float32)
        return self.net(x.reshape(-1, self.c, self.w, self.h)), state


cwd = os_path.dirname(__file__)
now = datetime.now()
timestamp = now.strftime("%m-%d-%Y_%H-%M-%S")
device = "cuda" if torch.cuda.is_available() else "cpu"

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


def get_env(render_mode='human'):
    """This function is needed to provide callables for DummyVectorEnv."""
    env = pklot_env.raw_env(render_mode=render_mode, n_vehicles=NUM_AGENT, random_reset=False, seed=1)
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
        net = CNNDQN(observation_space, device=device).to(device)

        actor = ActorProb(
            net, action_shape=env.action_space.shape, device=device, max_action=env.action_space.high[0]
        ).to(device)
        net2 = CNNDQN(observation_space, device=device).to(device)
        critic = Critic(net2, device=device).to(device)
        for m in set(actor.modules()).union(critic.modules()):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal_(m.weight)
                torch.nn.init.zeros_(m.bias)
        optim = torch.optim.Adam(
            set(actor.parameters()).union(critic.parameters()), lr=1e-5
        )

        def dist(*logits):
            try:
                return Independent(Normal(*logits), 1)
            except ValueError:
                pass
            # return TanhTransform(*logits)

        agents.append(
            PPOPolicy(
                actor=actor,
                critic=critic,
                optim=optim,
                discount_factor=0.9,
                dist_fn=dist,
                vf_coef=0.25,
                ent_coef=0.01,
                eps_clip=0.1,
                gae_lambda=0.95,
                max_grad_norm=0.25,
                action_scaling=True,
                reward_normalization=1,
                value_clip=True,
                advantage_normalization=True,
                recompute_advantage=False,
                action_space=env.action_space
            )  # .to(device)
        )

    policy = MultiAgentPolicyManager(agents, env, action_scaling=True,
                                     action_bound_method='clip')
    return policy, optim, env.agents


if __name__ == "__main__":
    # https://pettingzoo.farama.org/tutorials/tianshou/intermediate/
    # ======== Step 1: Environment setup =========
    # TODO: still don't quite get why we need dummy vectors
    train_env = SubprocVectorEnv([get_env for _ in range(16)])
    test_env = SubprocVectorEnv([get_env for _ in range(4)])

    # seed
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ======== Step 2: Agent setup =========
    policy, optim, agents = get_agents()

    # ======== Step 3: Collector setup =========
    buffer = VectorReplayBuffer(
        2000,
        buffer_num=len(train_env),
    )

    train_collector = Collector(
        policy,
        train_env,
        buffer,
    )
    test_collector = Collector(policy, test_env)


    # ======== Step 4: Callback functions setup =========

    def save_best_fn(policy):
        model_save_path = os.path.join("log", "ppo", "policy.pth")
        os.makedirs(os.path.join("log", "ppo"), exist_ok=True)
        for agent in agents:
            torch.save(policy.policies[agent].state_dict(), model_save_path)
            logger.wandb_run.save(model_save_path)
        render_ppo(agents, policy, get_env(render_mode='rgb_array'))
        logger.wandb_run.log({"video": wandb.Video("out.gif", fps=4, format="gif")})


    def stop_over_n(n=10):
        mean_n = deque(maxlen=n)

        def stop_fn(mean_rewards):
            # currently set to never stop
            mean_n.append(mean_rewards)
            return np.mean(mean_n) >= 70


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
            n_episode=1, render=1 / 30
        )
        rew = result["rews"].mean()
        print(f"Mean reward (over {result['n/ep']} episodes): {rew}")


    logger = WandbLogger(project="confrez-tianshou", name=f"ppo_cont{timestamp}", save_interval=4)
    script_path = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(script_path, f"log/ppo/run{timestamp}")
    writer = SummaryWriter(log_path)
    logger.load(writer)

    # ======== Step 5: Run the trainer =========
    result = onpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=1000,
        step_per_epoch=500,
        # step_per_collect=640,
        episode_per_collect=16,
        episode_per_test=4,
        batch_size=32,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_over_n(5),
        save_best_fn=save_best_fn,
        repeat_per_collect=2,
        test_in_train=False,
        reward_metric=reward_metric,
        logger=logger,
    )

    # return result, policy.policies[agents[1]]
    print(f"\n==========Result==========\n{result}")
    print("\n(the trained policy can be accessed via policy.policies[agents[1]])")
