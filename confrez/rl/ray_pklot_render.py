import argparse
import os

import numpy as np
import ray
import supersuit as ss
from PIL import Image
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.algorithms import Algorithm
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
import pklot_env_unicycle_cont as pklot_env_cont

os.environ["SDL_VIDEODRIVER"] = "dummy"

checkpoint_path = os.path.expanduser("ray_results/pk_lot/PPO-2-randFalse-m_cycles300"
                                     "/PPO_pk_lot_f1f50_00000_0_2023-09-02_02-49-01/checkpoint_000880")


def get_env(render=False):
    """This function is needed to provide callables for DummyVectorEnv."""
    env_config = pklot_env_cont.EnvParams(
        reward_stop=-1, reward_dist=-0.1, reward_time=-0.1, reward_heading=-0.1, reward_collision=-10, reward_goal=100, window_size=280
    )
    env = pklot_env_cont.raw_env(n_vehicles=2, random_reset=False, render_mode='rgb_array',
                                 params=env_config, max_cycles=200)

    env = ss.resize_v1(env, 84, 84)
    env = ss.black_death_v3(env)
    return env


env = get_env()
env_name = "pk_lot"
register_env(env_name, lambda config: PettingZooEnv(get_env()))

ray.init(local_mode=True)
PPO_agent = Algorithm.from_checkpoint(checkpoint_path)

reward_sum = 0
frame_list = []
i = 0
actions = {}
env.reset()

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    if termination or truncation:
        action = None
    else:
        action = PPO_agent.compute_single_action(observation.copy(), policy_id="shared_policy")

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
