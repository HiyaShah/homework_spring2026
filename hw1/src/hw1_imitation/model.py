"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn
import torch.nn.functional as F 


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        layers = []
        in_dim = state_dim 
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim 
        layers.append(nn.Linear(in_dim, chunk_size*action_dim))
        self.net = nn.Sequential(*layers)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        prediction = self.net(state).view(-1, self.chunk_size, self.action_dim)
        return F.mse_loss(prediction, action_chunk)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        with torch.no_grad():
            return self.net(state).view(-1, self.chunk_size, self.action_dim)


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        layers = []
        in_dim = state_dim + chunk_size*action_dim + 1
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim 
        layers.append(nn.Linear(in_dim, chunk_size*action_dim))
        self.net = nn.Sequential(*layers)
        
    def _predict_velocity(self, state: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor):
        batch_size = state.shape[0]
        x_flat = x_t.reshape(batch_size, -1)
        inp = torch.cat([state, x_flat, t], dim=1)
        return self.net(inp).view(-1,self.chunk_size, self.action_dim)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        device = state.device 
        batch_size = state.shape[0]
        t = torch.rand(batch_size, 1, device=device)
        noise = torch.randn(batch_size, self.chunk_size, self.action_dim, device=device)
        x_t = (1-t)[:, :, None]*noise+t[:, :, None]*action_chunk 
        v_true = action_chunk - noise 
        v_pred = self._predict_velocity(state, x_t, t)
        return F.mse_loss(v_pred, v_true)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        device = state.device 
        batch_size = state.shape[0]
        x = torch.randn(batch_size, self.chunk_size, self.action_dim, device = device)
        dt = 1.0 / num_steps
        with torch.no_grad():
            for k in range(num_steps):
                t = torch.full((batch_size, 1), k / num_steps, device=device)
                v = self._predict_velocity(state, x, t)
                x = x + v*dt 
        return x


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
