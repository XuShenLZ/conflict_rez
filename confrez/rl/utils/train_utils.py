import os
from typing import Callable, List, Tuple
import confrez.rl.envs.pklot_env as pklot_env
import confrez.rl.envs.pklot_env_unicycle as pklot_env_unicycle
import supersuit as ss
import torch
from scipy import interpolate
from model import CNNDQN, DuelingDQN
import gymnasium as gym

from tianshou.env import PettingZooEnv

from tianshou.policy import BasePolicy, DQNPolicy, MultiAgentPolicyManager


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


def get_env(render_mode="human", num_agents=4, seed=1, max_cycles=500):
    """This function is needed to provide callables for DummyVectorEnv."""
    env = pklot_env.raw_env(
        n_vehicles=num_agents,
        random_reset=False,
        seed=seed,
        max_cycles=max_cycles,
        render_mode=render_mode,
    )  # seed=1
    env = ss.black_death_v3(env)
    env = ss.resize_v1(env, 140, 140)
    return PettingZooEnv(env)


def get_unicycle_env(render_mode="human", num_agents=4, seed=1, max_cycles=500):
    """This function is needed to provide callables for DummyVectorEnv."""
    env = pklot_env_unicycle.raw_env(
        n_vehicles=num_agents,
        random_reset=False,
        seed=seed,
        max_cycles=max_cycles,
        render_mode=render_mode,
    )  # seed=1
    env = ss.black_death_v3(env)
    env = ss.resize_v1(env, 140, 140)
    return PettingZooEnv(env)


def get_agents(
    num_agents=4, discount_factor=0.9, estimation_step=4, action_shape=7
) -> Tuple[BasePolicy, List[torch.optim.Optimizer], list]:
    env = get_env()
    agents = []
    optims = []
    for _ in range(num_agents):
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
                discount_factor=discount_factor,
                estimation_step=estimation_step,
                target_update_freq=int(1000),
            ).to("cuda" if torch.cuda.is_available() else "cpu")
        )

    policy = MultiAgentPolicyManager(agents, env)
    return policy, optims, env.agents
