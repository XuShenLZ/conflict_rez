import argparse
import os
import random
from datetime import datetime

import pandas as pd
import numpy as np
from ray.rllib.env import ParallelPettingZooEnv

from ray.tune import run, sample_from, register_env
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.schedulers.pb2 import PB2
import pklot_env_unicycle_cont as pklot_env_cont
import ray


n_agents = 1
random_reset = False


# Postprocess the perturbed config to ensure it's still valid used if PBT.
def explore(config):
    # Ensure we collect enough timesteps to do sgd.
    if config["train_batch_size"] < config["sgd_minibatch_size"] * 2:
        config["train_batch_size"] = config["sgd_minibatch_size"] * 2
    # Ensure we run at least one sgd iter.
    if config["lambda"] > 1:
        config["lambda"] = 1
    config["train_batch_size"] = int(config["train_batch_size"])
    return config


if __name__ == "__main__":
    RAY_memory_monitor_refresh_ms = 0

    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=1000000)
    parser.add_argument("--algo", type=str, default="PPO")
    parser.add_argument("--num_workers", type=int, default=9)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--t_ready", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--horizon", type=int, default=500
    )  # make this 1000 for other envs
    parser.add_argument("--perturb", type=float, default=0.25)  # if using PBT
    parser.add_argument("--env_name", type=str, default="pk_lot")
    parser.add_argument(
        "--criteria", type=str, default="timesteps_total"
    )  # "training_iteration", "time_total_s"
    parser.add_argument(
        "--net", type=str, default="128_128"
    )  # May be important to use a larger network for bigger tasks.
    parser.add_argument("--filename", type=str, default="")
    parser.add_argument("--method", type=str, default="pb2")  # ['pbt', 'pb2']
    parser.add_argument("--save_csv", type=bool, default=True)

    args = parser.parse_args()


    def get_env(render=False):
        """This function is needed to provide callables for DummyVectorEnv."""
        env_config = pklot_env_cont.EnvParams(
            reward_stop=-10, reward_dist=1, reward_heading=0, reward_time=-1, reward_collision=-10, reward_goal=100,
            window_size=84
        )
        env = pklot_env_cont.parallel_env(n_vehicles=n_agents, random_reset=random_reset, render_mode="rgb_array",
                                          params=env_config, max_cycles=args.horizon, seed=args.seed)

        return env


    register_env("pk_lot", lambda config: ParallelPettingZooEnv(get_env()))
    env_name = "pk_lot"

    # # bipedalwalker needs 1600
    # if args.env_name in ["BipedalWalker-v2", "BipedalWalker-v3"]:
    #     args.horizon = 1600
    # else:
    #     args.horizon = 1000

    pbt = PopulationBasedTraining(
        time_attr=args.criteria,
        metric="episode_reward_mean",
        mode="max",
        perturbation_interval=args.t_ready,
        resample_probability=args.perturb,
        quantile_fraction=args.perturb,  # copy bottom % with top %
        # Specifies the search space for these hyperparams
        hyperparam_mutations={
            "lambda": lambda: random.uniform(0.9, 1.0),
            "clip_param": lambda: random.uniform(0.1, 0.5),
            "lr": lambda: random.uniform(1e-3, 1e-5),
            "train_batch_size": lambda: random.randint(1000, 40000),
            "num_sgd_iter": lambda: random.randint(5, 30),
            "sgd_minibatch_size": lambda: random.randint(32, 512),
            "vf_clip_param": lambda: random.randint(10, 100),
            "grad_clip": lambda: random.choice(np.logspace(-1, 1.3, 50)),
            "entropy_coeff": lambda: random.uniform(0, 1e-2),
            "vf_loss_coeff": lambda: random.uniform(0.1, 2),
        },
        custom_explore_fn=explore,
    )

    pb2 = PB2(
        time_attr=args.criteria,
        metric="episode_reward_mean",
        mode="max",
        perturbation_interval=args.t_ready,
        quantile_fraction=args.perturb,  # copy bottom % with top %
        # Specifies the hyperparam search space
        hyperparam_bounds={
            "lambda": [0.9, 1.0],
            "clip_param": [0.1, 0.5],
            "lr": [1e-5, 1e-3],
            "train_batch_size": [1000, 40000],
            "num_sgd_iter": [5, 30],
            "sgd_minibatch_size": [32, 512],
            "vf_clip_param": [10, 100],
            "grad_clip": [0.1, 20],
            "entropy_coeff": [0, 1e-2],
            "vf_loss_coeff": [0.1, 2],
        },
    )

    methods = {"pbt": pbt, "pb2": pb2}

    timelog = (
        str(datetime.date(datetime.now())) + "_" + str(datetime.time(datetime.now()))
    )

    args.dir = "{}_{}_{}_Size{}_{}_{}_{}".format(
        args.algo,
        args.filename,
        args.method,
        str(args.num_samples),
        args.env_name,
        args.criteria,
        args.max,
    )

    analysis = run(
        args.algo,
        name="{}_{}_{}_seed{}_{}".format(
            timelog, args.method, args.env_name, str(args.seed), args.filename
        ),
        scheduler=methods[args.method],
        verbose=1,
        num_samples=args.num_samples,
        reuse_actors=False,
        stop={args.criteria: args.max},
        config={
            "num_rollout_workers": args.num_workers,
            "env": args.env_name,
            "disable_env_checking": True,
            "log_level": "INFO",
            "seed": args.seed,
            "kl_coeff": 0.2,
            "num_gpus": 1 / args.num_samples,
            "horizon": args.horizon,
            "observation_filter": "MeanStdFilter",
            "model": {
                "dim": 84,
                # "fcnet_hiddens": [
                #     int(args.net.split("_")[0]),
                #     int(args.net.split("_")[1]),
                # ],
                "free_log_std": False,
                "framestack": True,
                # "conv_filters": [[16, [16, 16], 4], [32, [4, 4], 2], [64, [4, 4], 2], [512, [9, 9], 1]],
            },
            # "sgd_minibatch_size": 128,
            "num_sgd_iter": sample_from(lambda spec: random.randint(5, 30)),
            "sgd_minibatch_size": sample_from(lambda spec: random.randint(32, 512)),
            "kl_target": 1e-4,
            "vf_clip_param": sample_from(lambda spec: random.randint(10, 100)),
            "grad_clip": sample_from(lambda spec: random.choice(np.logspace(-1, 1.3, 50))),
            "lambda": sample_from(lambda spec: random.uniform(0.9, 1.0)),
            "clip_param": sample_from(lambda spec: random.uniform(0.1, 0.5)),
            "lr": sample_from(lambda spec: random.uniform(1e-3, 1e-5)),
            "entropy_coeff": sample_from(lambda spec: random.uniform(0, 1e-2)),
            "train_batch_size": sample_from(lambda spec: random.randint(1000, 40000)),
            "vf_loss_coeff": sample_from(lambda spec: random.uniform(0.1, 2)),
        },
        max_failures=-1
    )

    print(analysis.get_best_config('episode_reward_mean', 'max'))
    print(analysis.get_best_trial('episode_reward_mean', 'max').last_result)
    all_dfs = list(analysis.trial_dataframes.values())

    results = pd.DataFrame()
    for i in range(args.num_samples):
        df = all_dfs[i]
        df = df[
            [
                "timesteps_total",
                "episodes_total",
                "episode_reward_mean",
                # "info/learner/default_policy/cur_kl_coeff",
            ]
        ]
        df["Agent"] = i
        results = pd.concat([results, df]).reset_index(drop=True)

    if args.save_csv:
        if not (os.path.exists("data/" + args.dir)):
            os.makedirs("data/" + args.dir)

        results.to_csv("data/{}/seed{}.csv".format(args.dir, str(args.seed)))