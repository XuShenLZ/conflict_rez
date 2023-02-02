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

class CNN_DQN(Net): #TODO: nn.Module
    ## eps scheduler
    def __init__(self, 
                state_shape: Union[int, Sequence[int]], 
                action_shape: Union[int, Sequence[int]] = 0, 
                hidden_sizes: Sequence[int] = ..., 
                norm_layer: Optional[ModuleType] = None, 
                activation: Optional[ModuleType] = nn.ReLU, 
                device: Union[str, int, torch.device] = "cpu", 
                concat: bool = False, 
                softmax: bool = False, 
                num_atoms: int = 1, 
                dueling_param: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None, 
                linear_layer: Type[nn.Linear] = nn.Linear) -> None:
        super().__init__(state_shape, 
                        action_shape, 
                        hidden_sizes, 
                        norm_layer, 
                        activation, 
                        device, 
                        concat, 
                        softmax,
                        num_atoms, 
                        dueling_param, 
                        linear_layer)
        self.device = device
        self.num_atoms = num_atoms
        self.softmax = softmax
        input_dim = int(np.prod(state_shape))
        action_dim = int(np.prod(action_shape)) * num_atoms
        if concat:
            input_dim += action_dim
        self.use_dueling = dueling_param is not None
        output_dim = action_dim if not self.use_dueling and not concat else 0
        self.output_dim = self.model.output_dim
        if self.use_dueling:  # dueling DQN
            q_kwargs, v_kwargs = dueling_param  # type: ignore
            q_output_dim, v_output_dim = 0, 0
            if not concat:
                q_output_dim, v_output_dim = action_dim, num_atoms
            q_kwargs: Dict[str, Any] = {
                **q_kwargs, "input_dim": self.output_dim,
                "output_dim": q_output_dim,
                "device": self.device
            }
            v_kwargs: Dict[str, Any] = {
                **v_kwargs, "input_dim": self.output_dim,
                "output_dim": v_output_dim,
                "device": self.device
            }
            self.Q, self.V = MLP(**q_kwargs), MLP(**v_kwargs)
            self.output_dim = self.Q.output_dim

    def forward(
            self,
            obs: torch.Tensor,
            state: Any = None,
            info: Dict[str, Any] = {},
        ) -> Tuple[torch.Tensor, Any]:
            """Mapping: obs -> flatten (inside MLP)-> logits."""
            # print("DEBUG:", obs.shape)
            # obs_copy = obs.reshape((-1, 3, 140, 140))
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor()
            ])
            obs_tensor = []
            for o in obs:
                obs_tensor.append(transform(o))
            obs = torch.stack(tensors=obs_tensor, dim=0)
            # print("DEBUG:", obs.shape)
            # TODO: move to _init_
            self.cnn = nn.Sequential(
                #?? (10 3 140 140)
                nn.Conv2d(obs.shape[1], 32, kernel_size=8, stride=4, padding=0),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Flatten(),
            )
            with torch.no_grad():
                n_flatten = self.cnn(torch.as_tensor(obs).float()).shape[1]
            self.linear = nn.Sequential(nn.Linear(n_flatten, 7), nn.ReLU())
            # DEBUG: shape error
            logits = self.linear(self.cnn(obs))
            bsz = logits.shape[0]
            if self.use_dueling: 
                q, v = self.Q(logits), self.V(logits)
                if self.num_atoms > 1:
                    q = q.view(bsz, -1, self.num_atoms)
                    v = v.view(bsz, -1, self.num_atoms)
                logits = q - q.mean(dim=1, keepdim=True) + v
            elif self.num_atoms > 1:
                logits = logits.view(bsz, -1, self.num_atoms)
            if self.softmax:
                logits = torch.softmax(logits, dim=-1)
            return logits, state

class CNNDQN(nn.Module):
    def __init__(self, state_shape, action_shape, obs, device) -> None:
        super().__init__()
        self.device = device
        transform = torchvision.transforms.Compose([
                        torchvision.transforms.ToTensor()
                    ])
        self.cnn = nn.Sequential(
                #?? (10 3 140 140)
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
        self.linear = nn.Sequential(nn.Linear(n_flatten, action_shape), nn.ReLU())

    def forward(self, obs, state=None, info={}):
        transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor()
            ])
        obs_tensor = []
        for o in obs:
            obs_tensor.append(transform(o))
        obs = torch.stack(tensors=obs_tensor, dim=0)
        # print("DEBUG: obs shape", obs.shape)
        obs = obs.to(device = self.device)
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        logits = self.linear(self.cnn(obs))
        return logits, state
