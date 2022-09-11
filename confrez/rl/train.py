from typing import Callable, List
from confrez.rl.pklot_env import parallel_env
import supersuit as ss
import torch
from torch import nn, tensor
from stable_baselines3 import PPO, DQN
from stable_baselines3.dqn import CnnPolicy
from stable_baselines3.common import results_plotter
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from scipy import interpolate

from datetime import datetime
from os import path as os_path

import wandb
from wandb.integration.sb3 import WandbCallback
run = wandb.init(project="rl-parking", 
                entity="chengtianyue",
                monitor_gym=True,
                sync_tensorboard=True
                )

cwd = os_path.dirname(__file__)

MODEL_NAME = "DQN-Resnet-Extractor"

env = parallel_env(n_vehicles=4, seed=1, random_reset=True)
env = ss.black_death_v3(env)
env = ss.resize_v1(env, 140, 140)
# env = ss.color_reduction_v0(env, mode="B")
# env = ss.frame_stack_v2(env, 3)
env = ss.pettingzoo_env_to_vec_env_v1(env)
env = ss.concat_vec_envs_v1(env, 24, num_cpus=12, base_class="stable_baselines3")

now = datetime.now()
timestamp = now.strftime("%m-%d-%Y_%H-%M-%S")


def step_schedule(
    initial_value: float, steps: List[float], levels: List[float]
) -> Callable[[float], float]:
    """
    Step learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    weight_func = interpolate.interp1d(steps, levels, "next", fill_value="extrapolate")

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        weight = weight_func(progress_remaining).__float__()
        return weight * initial_value

    return func

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)

#https://www.kaggle.com/code/kwabenantim/gfootball-stable-baselines3
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.conv2 = conv3x3(out_channels, out_channels, stride)
        
    def forward(self, x):
        residual = x
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        return out
    
class ParkingCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        in_channels = observation_space.shape[0]  # channels x height x width
        self.cnn = nn.Sequential(
            conv3x3(in_channels=in_channels, out_channels=32),
            nn.MaxPool2d(kernel_size=3, stride=2, dilation=1, ceil_mode=False),
            ResidualBlock(in_channels=32, out_channels=32),
            ResidualBlock(in_channels=32, out_channels=32),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.linear = nn.Sequential(
          nn.Linear(in_features=152352, out_features=features_dim, bias=True),
          nn.ReLU(),
        )

    def forward(self, obs):
        return self.linear(self.cnn(obs))

policy_kwargs = dict(features_extractor_class=ParkingCNN,
                     features_extractor_kwargs=dict(features_dim=256))

model = DQN(
    CnnPolicy,
    env,
    policy_kwargs=policy_kwargs,
    learning_rate=step_schedule(0.0005, [1, 0.8, 0.6, 0.3], [1, 0.5, 0.1, 0.05]),
    verbose=0,
    buffer_size=100000,
    learning_starts=500,
    gamma=0.993,
    exploration_fraction=0.7,
    exploration_final_eps=0.2,
    tensorboard_log=f"{cwd}/DQN-CNN_tensorboard/",
)

model.learn(
    total_timesteps=150000000,
    tb_log_name=f"{MODEL_NAME}_{timestamp}",
    callback=WandbCallback(
        gradient_save_freq=100,
        model_save_path=f"models/{run.id}",
        model_save_freq=3000,
        verbose=2,
        log = "all"
    )
)

model.save(f"{MODEL_NAME}_{timestamp}")
print("Training Finished")
