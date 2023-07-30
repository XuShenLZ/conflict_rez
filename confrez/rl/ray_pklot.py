from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env
from ray.rllib.env import ParallelPettingZooEnv, MultiAgentEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.appo import APPOConfig
from ray.tune.logger import pretty_print
from ray import tune, air

from ray.air.integrations.wandb import setup_wandb
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune.schedulers import PopulationBasedTraining

import ray

import pklot_env_unicycle_cont as pklot_env_cont
import pklot_env_unicycle as pklot_env_disc
import supersuit as ss
import os
import random

from torch import nn


def get_env(render=False):
    """This function is needed to provide callables for DummyVectorEnv."""
    env_config = pklot_env_cont.EnvParams(
        reward_stop=-5, reward_dist=-0.1, reward_heading=-0.5, reward_time=0, reward_collision=-10, reward_goal=100,
        eps=1e-3
    )
    env = pklot_env_cont.parallel_env(n_vehicles=2, random_reset=False, seed=1, render_mode='rgb_array',
                                      params=env_config, max_cycles=200)

    env = ss.black_death_v3(env)
    env = ss.resize_v1(env, 84, 84)
    return env


if __name__ == '__main__':
    ray.init(local_mode=False)

    register_env('pk_lot', lambda config: ParallelPettingZooEnv(get_env()))
    env_name = 'pk_lot'
    env = get_env()

    config = (
        PPOConfig()
        .environment(env=env_name, clip_actions=True)
        .rollouts(num_rollout_workers=8, rollout_fragment_length=128, num_envs_per_worker=4)
        .training(
            train_batch_size=4096,
            lr=1e-4,
            kl_coeff=1,
            kl_target=0.01,
            gamma=0.99,
            lambda_=0.95,
            use_gae=True,
            clip_param=0.2,
            grad_clip=5,
            entropy_coeff=0.0,
            vf_loss_coeff=5,
            vf_clip_param=32,
            sgd_minibatch_size=256,
            num_sgd_iter=30,
            model={"dim": 84,  "use_lstm": True, "vf_share_layers": True},
        )
        .debugging(log_level="INFO")
        .framework(framework="torch")
        .resources(num_gpus=1)
        .multi_agent(
            policies=env.possible_agents,
            policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id)
        )
    )

    results = tune.run(
        "PPO",
        name="PPO",
        verbose=3,
        stop={"timesteps_total": 5000000, "episode_reward_mean": 30},
        checkpoint_freq=10,
        local_dir="ray_results/" + env_name,
        config=config.to_dict(),
        # run_config=air.RunConfig(stop={"episode_reward_mean": 30}),
        callbacks=[WandbLoggerCallback(project='confrez-ray')],
    )
