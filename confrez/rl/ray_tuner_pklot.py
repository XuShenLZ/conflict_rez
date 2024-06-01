import pprint
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
from ray.tune.schedulers import PopulationBasedTraining

import pklot_env_unicycle_cont as pklot_env_cont
from ray.rllib.policy.sample_batch import SampleBatch
import pklot_env_unicycle as pklot_env_disc
import supersuit as ss

import os
import random
from typing import Dict, Tuple, List
from torch import nn

n_agents = 1
random_reset = True
max_cycles = 200


def explore(config):
    # ensure we collect enough timesteps to do sgd
    if config["train_batch_size"] < config["sgd_minibatch_size"] * 2:
        config["train_batch_size"] = config["sgd_minibatch_size"] * 2
    # ensure we run at least one sgd iter
    if config["num_sgd_iter"] < 1:
        config["num_sgd_iter"] = 1
    return config


hyperparam_mutations = {
    "gamma": lambda: random.uniform(0.9, 0.99),
    "lambda_": lambda: random.uniform(0.95, 0.995),
    "clip_param": lambda: random.uniform(0.1, 0.5),
    "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
    "kl_target": lambda: random.uniform(1e-3, 1e-2),
    "kl_coeff": [0.2, 0.1, 0.02, 0.05],
    "num_sgd_iter": lambda: random.randint(1, 20),
    "vf_clip_param": lambda: random.randint(2, 64),
    "vf_loss_coeff": lambda: random.uniform(0.1, 5),
    "entropy_coeff": [1e-2, 1e-3, 1e-4, 1e-1]
}

pbt = PopulationBasedTraining(
    time_attr="time_total_s",
    perturbation_interval=1200,
    resample_probability=0.25,
    # Specifies the mutations of these hyperparams
    hyperparam_mutations=hyperparam_mutations,
    custom_explore_fn=explore,
)


def get_env(render=False):
    """This function is needed to provide callables for DummyVectorEnv."""
    env_config = pklot_env_cont.EnvParams(
        reward_stop=-10, reward_dist=-1, reward_heading=-1, reward_time=-1, reward_collision=-10, reward_goal=1000,
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
    num_samples = 4
    rollout_workers = 7
    rollout_length = 25
    num_envs_per = 4
    num_gpus = 0.9 / num_samples

    batch_size = rollout_workers * rollout_length * num_envs_per * 5
    mini_batch = 4

    config = (
        PPOConfig()
        .environment(env="pk_lot", disable_env_checking=True)  # , env_task_fn=curriculum_fn)
        .rollouts(num_rollout_workers=rollout_workers, rollout_fragment_length=rollout_length,
                  num_envs_per_worker=num_envs_per)
        .training(
            train_batch_size=batch_size,
            lr=tune.choice([1e-3, 5e-4, 1e-4, 5e-5, 1e-5]),
            kl_coeff=tune.choice([0.2, 1, 0.1, 0.05]),
            kl_target=tune.choice([1e-4, 1e-3, 1e-2]),
            gamma=tune.choice([0.9, 0.99, 0.95]),
            lambda_=tune.choice([0.95, 0.995, 0.99, 0.97]),
            clip_param=tune.choice([0.2, 0.3, 0.4]),
            grad_clip=tune.choice([5, 10, 15, 20]),
            entropy_coeff=tune.choice([0, 1e-3, 1e-4, 1e-2, 1e-1]),
            vf_loss_coeff=tune.choice([1, 0.5, 0.1, 0.05, 0.01]),
            vf_clip_param=tune.choice([10, 20, 32, 64]),
            sgd_minibatch_size=batch_size // mini_batch,
            num_sgd_iter=tune.choice([5, 10, 15, 20]),
            model={"dim": 140, "use_lstm": False, "framestack": True,  # "post_fcnet_hiddens": [512, 512],
                   "vf_share_layers": False, "free_log_std": False,
                   "conv_filters": [[16, [16, 16], 4], [32, [4, 4], 2], [64, [4, 4], 2], [512, [9, 9], 1]]},
        )
        .debugging(log_level="INFO")
        .framework(framework="torch")
        .resources(num_gpus=num_gpus, num_cpus_per_worker=1)
        .multi_agent(
            policies={"shared_policy"},
            policy_mapping_fn=(lambda agent_id, episode, worker, **kwargs: "shared_policy")
        )
    )

    results = tune.Tuner(
        "PPO",
        tune_config=tune.TuneConfig(
            metric="episode_reward_mean",
            mode="max",
            scheduler=pbt,
            num_samples=num_samples,
            reuse_actors=False,
        ),
        run_config=air.RunConfig(
            name=f"PPO-{n_agents}-rand{random_reset}-m_cycles{max_cycles}-tuner",
            stop={"episode_reward_mean": 20, "timesteps_total": 20_000_000},
            local_dir="ray_results/" + env_name,
            callbacks=[WandbLoggerCallback(project="confrez-ray", log_config=True)],
            verbose=0,
            failure_config=air.FailureConfig(
                max_failures=-1,
            )
        ),
        param_space=config.to_dict(),
        # resume=True
    ).fit()

    best_result = results.get_best_result()

    print("Best performing trial's final set of hyperparameters:\n")
    pprint.pprint(
        {k: v for k, v in best_result.config.items() if k in hyperparam_mutations}
    )

    print("\nBest performing trial's final reported metrics:\n")

    metrics_to_print = [
        "episode_reward_mean",
        "episode_reward_max",
        "episode_reward_min",
        "episode_len_mean",
    ]
    pprint.pprint({k: v for k, v in best_result.metrics.items() if k in metrics_to_print})