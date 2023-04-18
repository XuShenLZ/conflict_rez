import torch
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.data import Batch, to_numpy
import PIL
import os
import confrez.rl.envs.pklot_env
import confrez.rl.envs.pklot_env_unicycle
import supersuit as ss
from tianshou.env import PettingZooEnv, DummyVectorEnv
from PIL import Image
import numpy as np
from model import DuelingDQN
from tianshou.policy import DQNPolicy, MultiAgentPolicyManager


def get_agents():
    env = get_env()
    agents = []
    for _ in range(4):
        net = DuelingDQN(
            state_shape=(10, 3, 140, 140),
            action_shape=7,
            obs=env.observation_space.sample()[None],
            device="cuda" if torch.cuda.is_available() else "cpu",
        ).to("cuda" if torch.cuda.is_available() else "cpu")

        optim = torch.optim.Adam(net.parameters(), lr=1e-4)  # , eps=1.5e-4

        agents.append(
            DQNPolicy(
                model=net,
                optim=optim,
                discount_factor=0.9,
                estimation_step=4,
                target_update_freq=int(5000),
            ).to("cuda" if torch.cuda.is_available() else "cpu")
        )

    policy = MultiAgentPolicyManager(agents, env)
    return policy, env.agents


def get_env(render_mode="human", n_vehicles=4):
    """This function is needed to provide callables for DummyVectorEnv."""
    env = pklot_env.raw_env(
        n_vehicles=n_vehicles,
        random_reset=False,
        seed=1,
        max_cycles=500,
        render_mode=render_mode,
    )  # seed=1
    env = ss.black_death_v3(env)
    env = ss.resize_v1(env, 140, 140)
    return PettingZooEnv(env)


def get_env_unicycle(render_mode="human", n_vehicles=4):
    """This function is needed to provide callables for DummyVectorEnv."""
    env = pklot_env_unicycle.raw_env(
        n_vehicles=n_vehicles,
        random_reset=False,
        seed=1,
        max_cycles=500,
        render_mode=render_mode,
    )  # seed=1
    env = ss.black_death_v3(env)
    env = ss.resize_v1(env, 140, 140)
    return PettingZooEnv(env)


def render_human(agents, policy, n_vehicles=4):
    for i, agent in enumerate(agents):
        # shouldn't be in this way when num_agents > 1!
        filename = os.path.join("log", "dqn", f"policy{i}.pth")
        policy.policies[agent].load_state_dict(
            torch.load(filename, map_location=torch.device("cuda"))
        )
    for agent in agents:
        policy.policies[agent].set_eps(0)
    collector = Collector(policy, env, exploration_noise=False)
    result = collector.collect(n_episode=1, render=1 / 30)
    rews, lens = result["rews"], result["lens"]
    print(f"Final reward: {rews[:].mean()}, length: {lens.mean()}")


def render(agents, policy, n_vehicles=4):
    env = get_env(render_mode="rgb_array", n_vehicles=n_vehicles)
    frame_list = []
    obs = env.reset()
    obs = np.array([obs["obs"]])
    done = False
    total_reward = 0
    while not done:
        for agent in agents:
            logits = policy.policies[agent].model(obs)
            q, _ = policy.policies[agent].compute_q_value(logits, mask=None)
            act = to_numpy(q.max(dim=1)[1])
            obs, reward, done, _, _ = env.step(act)
            obs = np.array([obs["obs"]])
            frame_list.append(PIL.Image.fromarray(env.render()))
            total_reward += np.sum(reward)

    env.close()
    frame_list[0].save(
        "out.gif", save_all=True, append_images=frame_list[1:], duration=100, loop=0
    )


def render_unicycle(agents, policy, n_vehicles=4):
    env = get_env_unicycle(render_mode="rgb_array", n_vehicles=n_vehicles)
    frame_list = []
    obs = env.reset()
    obs = np.array([obs["obs"]])
    done = False
    total_reward = 0
    while not done:
        for agent in agents:
            logits = policy.policies[agent].model(obs)
            q, _ = policy.policies[agent].compute_q_value(logits, mask=None)
            act = to_numpy(q.max(dim=1)[1])
            obs, reward, done, _, _ = env.step(act)
            obs = np.array([obs["obs"]])
            frame_list.append(PIL.Image.fromarray(env.render()))
            total_reward += np.sum(reward)

    env.close()
    frame_list[0].save(
        "out.gif", save_all=True, append_images=frame_list[1:], duration=100, loop=0
    )


if __name__ == "__main__":
    # render_human()
    env = DummyVectorEnv([get_env])
    policy, agents = get_agents()

    for i, agent in enumerate(agents):
        # shouldn't be in this way when num_agents > 1!
        filename = os.path.join("log", "dqn", f"policy{i}.pth")
        policy.policies[agent].load_state_dict(
            torch.load(filename, map_location=torch.device("cuda"))
        )
    render(agents, policy)
