from abc import ABC

from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env
from ray.rllib.env import ParallelPettingZooEnv, MultiAgentEnv
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv, TaskType
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.a2c import A2CConfig
from ray.rllib.algorithms.td3 import TD3Config
from ray.rllib.algorithms.sac import SACConfig
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

n_agents = 2
random_reset = True
max_cycles = 200


class CNNModelV2(TorchModelV2, nn.Module):
    def __init__(self, obs_space, act_space, num_outputs, *args, **kwargs):
        TorchModelV2.__init__(self, obs_space, act_space, num_outputs, *args, **kwargs)
        nn.Module.__init__(self)
        self.model = nn.Sequential(
            nn.Conv2d(12, 32, [8, 8], stride=(4, 4)),
            nn.ReLU(),
            nn.Conv2d(32, 64, [4, 4], stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 64, [3, 3], stride=(1, 1)),
            nn.ReLU(),
            nn.Flatten(),
            (nn.Linear(3136, 512)),
            nn.ReLU(),
        )
        self.policy_fn = nn.Linear(512, num_outputs)
        self.value_fn = nn.Linear(512, 1)

    def forward(self, input_dict, state, seq_lens):
        model_out = self.model(input_dict["obs"].permute(0, 3, 1, 2))
        self._value_out = self.value_fn(model_out)
        return self.policy_fn(model_out), state

    def value_function(self):
        return self._value_out.flatten()


def get_env(render=False):
    """This function is needed to provide callables for DummyVectorEnv."""
    env_config = pklot_env_cont.EnvParams(
        reward_stop=-5, reward_dist=-0.1, reward_heading=-0.5, reward_time=0, reward_collision=-10, reward_goal=100,
        # eps=1e-3
    )
    env = pklot_env_cont.parallel_env(n_vehicles=n_agents, random_reset=random_reset, render_mode="rgb_array",
                                      params=env_config, max_cycles=max_cycles)

    # env = ss.sticky_actions_v0(env, repeat_action_probability=0.05)
    # env = ss.frame_skip_v0(env, num_frames=1)
    # env = ss.color_reduction_v0(env)
    # env = ss.frame_stack_v2(env)
    env = ss.black_death_v3(env)
    env = ss.clip_actions_v0(env)
    env = ss.resize_v1(env, 84, 84)

    return env


if __name__ == "__main__":
    # Reward normalization: TODO
    ray.init(local_mode=False)

    # ModelCatalog.register_custom_model("CNNModelV2", CNNModelV2)
    register_env("pk_lot", lambda config: ParallelPettingZooEnv(get_env()))
    env_name = "pk_lot"
    env = get_env()
    rollout_workers = 12
    rollout_length = 100
    num_envs_per = 4

    batch_size = rollout_workers * rollout_length * num_envs_per
    mini_batch = 4

    config = (
        PPOConfig()
        .environment(env="pk_lot", clip_actions=True)  # , env_task_fn=curriculum_fn)
        .rollouts(num_rollout_workers=rollout_workers, rollout_fragment_length=rollout_length,
                  num_envs_per_worker=num_envs_per, observation_filter="MeanStdFilter")
        .training(
            train_batch_size=batch_size,
            lr=5e-4,
            kl_coeff=1,
            kl_target=0.01,
            gamma=0.99,
            lambda_=0.95,
            use_gae=True,
            clip_param=0.4,
            grad_clip=10,
            entropy_coeff=0,
            vf_loss_coeff=0.2,
            vf_clip_param=5,
            sgd_minibatch_size=batch_size // mini_batch,
            num_sgd_iter=10,
            model={"dim": 84, "use_lstm": True, "framestack": False, "fcnet_hiddens": [512, 512],
                   "vf_share_layers": False, "free_log_std": True, "fcnet_activation": "relu"},
        )
        .debugging(log_level="INFO")
        .framework(framework="torch")
        .resources(num_gpus=1)
        .multi_agent(
            policies={"shared_policy"},
            policy_mapping_fn=(lambda agent_id, episode, worker, **kwargs: "shared_policy")
        )
    )

    # config = (
    #     A2CConfig()
    #     .environment(env="pk_lot", clip_actions=True)  # , env_task_fn=curriculum_fn)
    #     .rollouts(num_rollout_workers=rollout_workers, rollout_fragment_length=rollout_length,
    #               num_envs_per_worker=num_envs_per, observation_filter="MeanStdFilter")
    #     .training(
    #         train_batch_size=batch_size,
    #         grad_clip=30,
    #         model={"dim": 84, "use_lstm": True, "framestack": True, "fcnet_activation": "relu"},
    #     )
    #     .debugging(log_level="INFO")
    #     .framework(framework="torch")
    #     .resources(num_gpus=1)
    #     .multi_agent(
    #         policies=env.possible_agents,
    #         policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id)
    #     )
    # )

    # config = (
    #     SACConfig()
    #     .environment(env="pk_lot", normalize_actions=True)
    #     .rollouts(num_rollout_workers=rollout_workers, rollout_fragment_length=rollout_length,
    #               num_envs_per_worker=num_envs_per, observation_filter="MeanStdFilter")
    #     .training(
    #         train_batch_size=batch_size,
    #         # grad_clip=5,
    #         n_step=3,
    #         num_steps_sampled_before_learning_starts=10000,
    #         optimization_config={
    #             "actor_learning_rate": 3e-4,
    #             "critic_learning_rate": 3e-5,
    #             "entropy_learning_rate": 3e-4,
    #         },
    #         replay_buffer_config={
    #             "_enable_replay_buffer_api": True,
    #             "type": "MultiAgentPrioritizedReplayBuffer",
    #             "capacity": int(5e5),
    #             "prioritized_replay": False,
    #             "prioritized_replay_alpha": 0.6,
    #             "prioritized_replay_beta": 0.4,
    #             "prioritized_replay_eps": 1e-6,
    #             "worker_side_prioritization": False,
    #         },
    #         target_network_update_freq=1
    #     )
    #     .debugging(log_level="INFO")
    #     .framework(framework="torch")
    #     .resources(num_gpus=1)
    #     .multi_agent(
    #         policies=env.possible_agents,
    #         policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id)
    #     )
    #     .reporting(
    #         min_sample_timesteps_per_iteration=1000,
    #         metrics_num_episodes_for_smoothing=5
    #     )
    # )

    results = tune.run(
        "PPO",
        name=f"PPO-{n_agents}-rand{random_reset}-m_cycles{max_cycles}",
        verbose=0,
        metric="episode_reward_mean",
        mode="max",
        stop={"episode_reward_mean": 20},
        checkpoint_freq=10,
        local_dir="ray_results/" + env_name,
        config=config.to_dict(),
        callbacks=[WandbLoggerCallback(project="confrez-ray")],
        resume="AUTO"
    )
