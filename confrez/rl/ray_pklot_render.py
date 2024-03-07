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

checkpoint_path = os.path.expanduser("ray_results/pk_lot/PPO-4-randFalse-m_cycles500/PPO_pk_lot_f2d38_00000_0_2023-10-22_15-54-25/checkpoint_004730")


def get_env(render=False):
    """This function is needed to provide callables for DummyVectorEnv."""
    env_config = pklot_env_cont.EnvParams(
        reward_stop=-1, reward_dist=-0.1, reward_heading=-0.1, reward_time=-0.1, reward_collision=-1, reward_goal=100,
        window_size=140
    )
    env = pklot_env_cont.parallel_env(n_vehicles=4, random_reset=True, render_mode="rgb_array",
                                      params=env_config, max_cycles=1000)
    return env


env = get_env()
env_name = "pk_lot"
register_env(env_name, lambda config: ParallelPettingZooEnv(get_env()))

ray.init(local_mode=True, num_gpus=1)
PPO_agent = Algorithm.from_checkpoint(checkpoint_path)  #{agent_id: Policy.from_checkpoint(f'{checkpoint_path}/policies/{agent_id}') for agent_id in env.possible_agents}
# PPO_agent = PPOConfig().environment('pk_lot', disable_env_checking=True).training(model={"dim": 140, "use_lstm": False, "framestack": True,  # "post_fcnet_hiddens": [512, 512],
#                    "vf_share_layers": True, "free_log_std": False,
#                    "conv_filters": [[16, [16, 16], 4], [32, [4, 4], 2], [64, [4, 4], 2], [512, [9, 9], 1]]},).build()
# PPO_agent.restore(checkpoint_path)

reward_sum = 0
frame_list = []
obs_list = []
i = 0
actions = {}
obs, _ = env.reset()

while True:
    actions = {}
    for num in range(env.num_agents):
        agent = list(obs.keys())[num]
        current_obs = obs[agent].copy()
        action = (PPO_agent.compute_single_action
                           (current_obs, policy_id=agent))
        action = np.clip(action, env.action_space(agent).low, env.action_space(agent).high)
        actions[agent] = action
    obs, reward, termination, truncation, _ = env.step(actions)
    if False not in termination.values():
        break

    reward_sum += sum(reward.values())

    i += 1
    if i % (len(env.possible_agents) + 1) == 0:
        img = Image.fromarray(env.render())
        frame_list.append(img)
        # img = Image.fromarray(obs)
        # obs_list.append(img)
env.close()

print(reward_sum)
frame_list[0].save(
    "out.gif", save_all=True, append_images=frame_list[1:], duration=3, loop=0
)
