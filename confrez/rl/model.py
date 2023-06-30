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
import torch.nn.functional as F
import gymnasium as gym
from tianshou.utils.net.common import Net, MLP
import tianshou

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
            ResBlock(64, 64, kernel_size=3),
            ResBlock(64, 64, kernel_size=3),
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
            obs_transformed = obs_transformed.reshape(state_shape)
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
            obs_transformed = obs_transformed.reshape(state_shape)
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


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding='same')
        )

    def forward(self, x):
        res = x
        out = self.conv(x)

        out += res
        return F.relu(out)


class CriticCont(nn.Module):
    def __init__(
        self,
        state_shape,
        action_shape,
        device: Union[str, int, torch.device] = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.conv = nn.Sequential(
            nn.Conv2d(state_shape[1], 32, kernel_size=8, stride=4), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True),
            nn.Flatten()
        ).to(device)
        with torch.no_grad():
            dim = np.prod(self.conv(torch.zeros(state_shape).to(device)).shape[1:])
        self.lin1 = nn.Linear(dim, 256)
        self.last = nn.Sequential(
            nn.Linear(256 + action_shape, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        ).to(device)

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        act: Optional[Union[np.ndarray, torch.Tensor]] = None,
        info: Dict[str, Any] = {},
    ) -> torch.Tensor:
        """Mapping: (s, a) -> logits -> Q(s, a)."""
        if type(obs) is tianshou.data.Batch:
            obs = obs['obs']
        # obs = obs.reshape(-1, 140, 140, 3)
        if type(obs) is np.ndarray:
            obs = torch.tensor(obs, dtype=torch.float).to(self.device)
        else:
            pass
        obs = torch.transpose(obs, 1, 3).to(self.device)
        if type(act) is np.ndarray:
            act = torch.tensor(act, dtype=torch.float).to(self.device)
        logits = self.conv(obs)
        logits = F.relu(self.lin1(logits))
        logits = torch.cat((logits, act), 1)
        logits = self.last(logits)
        return logits