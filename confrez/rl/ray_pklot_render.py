import argparse
import os

import numpy as np
import ray
import supersuit as ss
from PIL import Image
from ray.rllib.algorithms import Algorithm
from ray.rllib.policy.policy import Policy
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv, ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.tune import tune
from ray.tune.registry import register_env
import pklot_env_unicycle_cont as pklot_env_cont
import matplotlib.pyplot as plt
from ray.rllib.algorithms.ppo import PPO, PPOConfig

os.environ["SDL_VIDEODRIVER"] = "dummy"

checkpoint_path = os.path.expanduser("ray_results/pk_lot/PPO-2-randTrue-m_cycles500/PPO_pk_lot_6a56d_00000_0_2024-09-03_03-51-03/checkpoint_001160")


def get_env(render=False):
    """This function is needed to provide callables for DummyVectorEnv."""
    env_config = pklot_env_cont.EnvParams(
        reward_stop=-1, reward_dist=10, reward_heading=0, reward_time=-1, reward_collision=-1, reward_goal=1000,
    )
    env = pklot_env_cont.parallel_env(n_vehicles=2, random_reset=True, render_mode="rgb_array", seed=0,
                                      params=env_config, max_cycles=500, return_scaled=True, resize=(140, 140))
    return env


env = get_env()
env_name = "pk_lot"
register_env(env_name, lambda config: ParallelPettingZooEnv(get_env()))

ray.init(local_mode=True, num_gpus=1)
PPO_agent = Algorithm.from_checkpoint(checkpoint_path)

reward_sum = 0
frame_list = []
obs_list = []
i = 0
actions = {}
env.reset()
obs, _ = env.reset()
# print(obs['vehicle_0'].shape)
# img = Image.fromarray((obs['vehicle_0'] * 256).astype(np.uint8))
# img.save('temp0.png')
# img = Image.fromarray((obs['vehicle_1'] * 256).astype(np.uint8))
# img.save('temp1.png')

# exit(0)
# print(env.render().shape)

while True:
    actions = {}
    for agent in env.agents:
        # agent = list(obs.keys())[num]
        current_obs = obs[agent].copy()
        action = (PPO_agent.compute_single_action
                           (current_obs, policy_id="shared_policy"))
        action = np.clip(action, env.action_space(agent).low, env.action_space(agent).high)
        actions[agent] = action
    obs, reward, termination, truncation, _ = env.step(actions)
    if False not in termination.values():
        break

    reward_sum += sum(reward.values())
    num_collisions = sum([r < -5 for r in reward.values()])

    i += 1
    if i % (len(env.possible_agents) + 1) == 0:
        img = Image.fromarray(env.render())
        frame_list.append(img)
        # img = Image.fromarray(obs)
        # obs_list.append(img)
env.close()

print(reward_sum)
print(num_collisions)
frame_list[0].save(
    f"{reward_sum}.gif", save_all=True, append_images=frame_list[1:], duration=3, loop=0
)
