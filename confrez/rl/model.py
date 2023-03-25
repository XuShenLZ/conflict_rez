from typing import (
    Any,
    Dict,
    Optional,
    List,
    Callable,
    Sequence,
    Tuple,
    Type,
    Union,
    no_type_check,
)
import numpy as np
import torch
import torchvision
from torch import nn
import gymnasium as gym
from tianshou.utils.net.common import Net, MLP

ModuleType = Type[nn.Module]


class CNNDQN(nn.Module):
    def __init__(self, state_shape, action_shape, obs, device, feature_dim=512) -> None:
        super().__init__()
        self.device = device
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        self.cnn = nn.Sequential(
            # ?? (10 3 140 140)
            nn.Conv2d(state_shape[1], 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            for o in obs:
                obs_transformed = transform(o)
            obs_transformed = obs_transformed.reshape(1, 3, 140, 140)
            # print("DEBUG: obs shape", obs_transformed.shape)
            after_cnn = self.cnn(obs_transformed.float())
            # print("DEBUG: after cnn shape:", after_cnn.shape)
            n_flatten = after_cnn.shape[1]
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, action_shape),
        )

    def forward(self, obs, state=None, info={}):
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        obs_tensor = []
        for o in obs:
            if type(o) is np.ndarray:
                obs_tensor.append(transform(o))
            else:
                obs_tensor.append(transform(o.obs))
        obs = torch.stack(tensors=obs_tensor, dim=0)
        # print("DEBUG: obs shape", obs.shape)
        obs = obs.to(device=self.device)
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        logits = self.linear(self.cnn(obs))
        return logits, state


class DuelingDQN(nn.Module):
    def __init__(self, state_shape, action_shape, obs, device, feature_dim=512) -> None:
        super().__init__()
        self.device = device
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        self.cnn = nn.Sequential(
            # ?? (10 3 140 140)
            nn.Conv2d(state_shape[1], 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            for o in obs:
                obs_transformed = transform(o)
            obs_transformed = obs_transformed.reshape(1, 3, 140, 140)
            # print("DEBUG: obs shape", obs_transformed.shape)
            after_cnn = self.cnn(obs_transformed.float())
            # print("DEBUG: after cnn shape:", after_cnn.shape)
            n_flatten = after_cnn.shape[1]
        self.advantage = nn.Sequential(
            nn.Linear(n_flatten, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, action_shape)
        )
        self.val = nn.Sequential(
            nn.Linear(n_flatten, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1)
        )

    def forward(self, obs, state=None, info={}):
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        obs_tensor = []
        for o in obs:
            if type(o) is np.ndarray:
                obs_tensor.append(transform(o))
            else:
                obs_tensor.append(transform(o.obs))
        obs = torch.stack(tensors=obs_tensor, dim=0)
        # print("DEBUG: obs shape", obs.shape)
        obs = obs.to(device=self.device)
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        feat = self.cnn(obs)
        val = self.val(feat)
        advantage = self.advantage(feat)
        logits = val + (advantage + torch.mean(advantage.detach(), dim=0))
        return logits, state


class Actor(nn.Module):
    def __init__(self, state_shape, action_shape, obs, device, feature_dim=512) -> None:
        super().__init__()
        self.device = device
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        self.cnn = nn.Sequential(
            nn.Conv2d(state_shape[1], 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            for o in obs:
                obs_transformed = transform(o)
            obs_transformed = obs_transformed.reshape(1, 3, 140, 140)
            after_cnn = self.cnn(obs_transformed.float())
            n_flatten = after_cnn.shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, action_shape),
            nn.Softmax(dim=-1)
        )

    def forward(self, obs, state=None, info={}):
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        obs_tensor = []
        for o in obs:
            if type(o) is np.ndarray:
                obs_tensor.append(transform(o))
            else:
                obs_tensor.append(transform(o.obs))
        obs = torch.stack(tensors=obs_tensor, dim=0)
        obs = obs.to(device=self.device)
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        state = self.cnn(obs)
        logits = self.linear(state)
        return logits, state


class Critic(nn.Module):
    def __init__(self, state_shape, obs, device, feature_dim=512) -> None:
        super().__init__()
        self.device = device
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        self.cnn = nn.Sequential(
            nn.Conv2d(state_shape[1], 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            for o in obs:
                obs_transformed = transform(np.array(o))
            obs_transformed = obs_transformed.reshape(1, 3, 140, 140)
            after_cnn = self.cnn(obs_transformed.float())
            n_flatten = after_cnn.shape[1]
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1),
        )

    def forward(self, obs, state=None, info={}):
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        obs_tensor = []
        for o in obs:
            if type(o) is np.ndarray:
                obs_tensor.append(transform(o))
            else:
                obs_tensor.append(transform(o.obs))
        obs = torch.stack(tensors=obs_tensor, dim=0)
        obs = obs.to(device=self.device)
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        logits = self.linear(self.cnn(obs))
        return logits
