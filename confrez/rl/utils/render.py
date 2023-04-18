from tianshou.data import to_numpy
import PIL
import confrez.rl.envs.pklot_env as pklot_env
import confrez.rl.envs.pklot_env_unicycle as pklot_env_unicycle
import supersuit as ss
from tianshou.env import PettingZooEnv
import numpy as np


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
