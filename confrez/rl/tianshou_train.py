import os
from typing import Callable, List, Optional, Tuple
import pklot_env
from pklot_env import parallel_env
import supersuit as ss
from datetime import datetime
from os import path as os_path
import torch
import numpy as np
from model import CNN_DQN

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv, PettingZooEnv

from tianshou.policy import (
    BasePolicy,
    DQNPolicy,
    MultiAgentPolicyManager
)
from tianshou.trainer import offpolicy_trainer
from tianshou.utils.net.common import Net

cwd = os_path.dirname(__file__)
now = datetime.now()
timestamp = now.strftime("%m-%d-%Y_%H-%M-%S")

MODEL_NAME = "Tianshou-Multiagent"
NUM_AGENT = 4

def get_env():
    """This function is needed to provide callables for DummyVectorEnv."""
    env = pklot_env.raw_env(n_vehicles=4, seed=1, random_reset=False)
    env = ss.black_death_v3(env)
    env = ss.resize_v1(env, 140, 140)
    return PettingZooEnv(env)

def get_agents(
    optim: Optional[torch.optim.Optimizer] = None,
) -> Tuple[BasePolicy, torch.optim.Optimizer, list]:
    env = get_env()
    agents = []
    for _ in range(NUM_AGENT):
        net = CNN_DQN(
            state_shape=58800,
            action_shape=7, #see pklot_env line 99
            hidden_sizes=[128, 128, 128, 128],
            device="cuda" if torch.cuda.is_available() else "cpu",
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        if optim is None:
            optim = torch.optim.Adam(net.parameters(), lr=1e-4)
        
        agents.append(DQNPolicy(
            model=net,
            optim=optim,
            discount_factor=0.9,
            estimation_step=100,
            target_update_freq=320,
        ))

    policy = MultiAgentPolicyManager(agents, env)
    return policy, optim, env.agents

if __name__ == "__main__":
    #https://pettingzoo.farama.org/tutorials/tianshou/intermediate/
    # ======== Step 1: Environment setup =========
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
        VectorReplayBuffer(20_000, len(train_env)),
        exploration_noise=True,
    )
    test_collector = Collector(policy, test_env, exploration_noise=True)
    # policy.set_eps(1)
    train_collector.collect(n_step=64 * 10)  # batch size * training_num

    # ======== Step 4: Callback functions setup =========
    def save_best_fn(policy):
        model_save_path = os.path.join("log", "rps", "dqn", "policy.pth")
        os.makedirs(os.path.join("log", "rps", "dqn"), exist_ok=True)
        torch.save(policy.policies[agents[1]].state_dict(), model_save_path)

    def stop_fn(mean_rewards):
        return mean_rewards >= 0.6

    def train_fn(epoch, env_step):
        policy.policies[agents[1]].set_eps(0.1)

    def test_fn(epoch, env_step):
        policy.policies[agents[1]].set_eps(0.05)

    def reward_metric(rews):
        return rews[:, 1]

    # ======== Step 5: Run the trainer =========
    result = offpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=50,
        step_per_epoch=1000,
        step_per_collect=50,
        episode_per_test=10,
        batch_size=64,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        update_per_step=0.1,
        test_in_train=False,
        reward_metric=reward_metric,
    )

    # return result, policy.policies[agents[1]]
    print(f"\n==========Result==========\n{result}")
    print("\n(the trained policy can be accessed via policy.policies[agents[1]])")
