from abc import ABC

import numpy as np
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env
from ray.rllib.env import ParallelPettingZooEnv, MultiAgentEnv, PettingZooEnv
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv, TaskType
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from ray.rllib.algorithms.sac.sac_torch_policy import SACTorchPolicy
from ray.rllib.algorithms.sac.sac_torch_model import SACTorchModel
from ray import tune, air

from ray.air.integrations.wandb import setup_wandb
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy

import ray

import pklot_env_unicycle_cont as pklot_env_cont
from ray.rllib.policy.sample_batch import SampleBatch
import pklot_env_unicycle as pklot_env_disc
import supersuit as ss

import os
import random
from typing import Dict, Tuple, List
from torch import nn

n_agents = 4
random_reset = False
max_cycles = 500


def get_env(render=False):
    """This function is needed to provide callables for DummyVectorEnv."""
    env_config = pklot_env_cont.EnvParams(
        reward_stop=-1, reward_dist=-0.1, reward_heading=-0.1, reward_time=-0.1, reward_collision=-1, reward_goal=100,
        window_size=140
    )
    env = pklot_env_cont.parallel_env(n_vehicles=n_agents, random_reset=random_reset, render_mode="rgb_array",
                                      params=env_config, max_cycles=max_cycles)

    return env


if __name__ == "__main__":
    # Reward normalization: TODO
    ray.init(local_mode=False)

    # ModelCatalog.register_custom_model("CNNModelV2", CNNModelV2)
    register_env("pk_lot", lambda config: ParallelPettingZooEnv(get_env()))
    env_name = "pk_lot"
    env = get_env()
    rollout_workers = 10
    rollout_length = 100
    num_envs_per = 2

    batch_size = rollout_workers * rollout_length * num_envs_per
    mini_batch = 8

    config = (
        PPOConfig()  # Version 2.5.0
        .environment(env="pk_lot", disable_env_checking=True, render_env=False)  # , env_task_fn=curriculum_fn
        .rollouts(num_rollout_workers=rollout_workers, rollout_fragment_length=rollout_length,
                  num_envs_per_worker=num_envs_per)
        .training(
            train_batch_size=batch_size,
            lr=5e-4,
            kl_coeff=0.2,
            kl_target=1e-3,
            gamma=0.99,
            lambda_=0.95,
            use_gae=True,
            clip_param=0.3,
            grad_clip=20,
            entropy_coeff=1e-2,
            vf_loss_coeff=0.05,  # 0.05
            vf_clip_param=10,  # 10 (2 vehicle)
            sgd_minibatch_size=512,
            num_sgd_iter=20,
            model={"dim": 140, "use_lstm": False, "framestack": True,  # "post_fcnet_hiddens": [512, 512],
                   "vf_share_layers": True, "free_log_std": False,
                   "conv_filters": [[16, [16, 16], 4], [32, [4, 4], 2], [64, [4, 4], 2], [512, [9, 9], 1]]},
        )
        .debugging(log_level="INFO")
        .framework(framework="torch")
        .resources(num_gpus=1)
        .multi_agent(
            policies=env.possible_agents,  # {"shared_policy"},
            policy_mapping_fn=(lambda agent_id, episode, worker, **kwargs: agent_id)  # "shared_policy")
        )
    )

    results = tune.run(
        "PPO",
        name=f"PPO-{n_agents}-rand{random_reset}-m_cycles{max_cycles}",
        # + "/PPO_pk_lot_28df8_00000_0_2023-09-11_14-50-01",
        verbose=0,
        metric="episode_reward_mean",
        mode="max",
        stop={"episode_reward_mean": 20},
        checkpoint_freq=10,
        local_dir="ray_results/" + env_name,
        config=config.to_dict(),
        max_failures=-1,
        callbacks=[WandbLoggerCallback(project="confrez-ray", entity="confrez")],
    )
