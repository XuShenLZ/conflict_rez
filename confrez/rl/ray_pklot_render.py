import argparse
import os

import numpy as np
import ray
import supersuit as ss
from PIL import Image
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
import pklot_env_unicycle_cont as pklot_env_cont

from pettingzoo.butterfly import pistonball_v6

os.environ["SDL_VIDEODRIVER"] = "dummy"

checkpoint_path = os.path.expanduser('ray_results/pk_lot/PPO/PPO_pk_lot_dbc48_00000_0_2023-07-28_23-48-00/'
                                     'checkpoint_000400')


def get_env(render=False):
    """This function is needed to provide callables for DummyVectorEnv."""
    env_config = pklot_env_cont.EnvParams(
        reward_stop=-5, reward_dist=-0.1, reward_time=0, reward_collision=-10, reward_goal=100
    )
    env = pklot_env_cont.raw_env(n_vehicles=2, random_reset=False, seed=1, render_mode='rgb_array',
                                 params=env_config, max_cycles=200)

    env = ss.black_death_v3(env)
    env = ss.resize_v1(env, 84, 84)
    return env


env = get_env()
env_name = "pk_lot"
register_env(env_name, lambda config: PettingZooEnv(get_env()))

ray.init()

PPOagent = PPO.from_checkpoint(checkpoint_path)

reward_sum = 0
frame_list = []
i = 0
state = [np.zeros([256], np.float32) for _ in range(2)]
actions = {}
env.reset()

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    if termination or truncation:
        action = None
    else:
        action, state, _ = PPOagent.compute_single_action(observation, state)

    env.step(action)

    reward_sum += reward

    i += 1
    if i % (len(env.possible_agents) + 1) == 0:
        img = Image.fromarray(env.render())
        frame_list.append(img)
env.close()

print(reward_sum)
frame_list[0].save(
    "out.gif", save_all=True, append_images=frame_list[1:], duration=3, loop=0
)
