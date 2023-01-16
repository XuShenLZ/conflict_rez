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
from torch import nn
import gymnasium as gym
from tianshou.policy import DQNPolicy

Schedule = Callable[[float], float]

class BaseFeaturesExtractor(nn.Module):
    """
    Adapted from SB3
    """

    def __init__(self, observation_space: gym.Space, features_dim: int = 0):
        super().__init__()
        assert features_dim > 0
        self._observation_space = observation_space
        self._features_dim = features_dim

    @property
    def features_dim(self) -> int:
        return self._features_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

class NatureCNN(BaseFeaturesExtractor):
    """
    Adapted from SB3
    CNN from DQN Nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

class CNNPolicy(DQNPolicy):
    def __init__(self, 
                model: torch.nn.Module, 
                optim: torch.optim.Optimizer, 
                observation_space: gym.spaces.Space,
                action_space: gym.spaces.Space,
                lr_schedule: Schedule,
                net_arch: Optional[List[int]] = None,
                activation_fn: Type[nn.Module] = nn.ReLU,
                features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
                features_extractor_kwargs: Optional[Dict[str, Any]] = None,
                normalize_images: bool = True,
                discount_factor: float = 0.99, 
                estimation_step: int = 1, 
                target_update_freq: int = 0, 
                reward_normalization: bool = False, 
                is_double: bool = True, 
                clip_loss_grad: bool = False,
                **kwargs: Any) -> None:
        super().__init__(model, 
                        optim, 
                        discount_factor, 
                        estimation_step, 
                        target_update_freq, 
                        reward_normalization, 
                        is_double, 
                        clip_loss_grad, 
                        **kwargs)
        
        
