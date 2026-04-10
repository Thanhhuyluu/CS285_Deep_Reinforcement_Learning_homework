"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


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
        self.state_dim = state_dim 
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.hidden_dims = hidden_dims
        self.layers = []
        for h in hidden_dims:
            self.layers.append(nn.Linear(state_dim, h))
            self.layers.append(nn.ReLU())
            state_dim = h

        self.layers.append(nn.Linear(state_dim, chunk_size * action_dim))

        self.model = nn.Sequential(
            *self.layers
        ) 

    def compute_loss(
        self,
        state: torch.Tensor, # (B, state_dim)
        action_chunk: torch.Tensor, # (B, chunk_size, action_dim)
    ) -> torch.Tensor:
        batch_size = state.shape[0] 
        flat = self.model(state)                                             # (B, chunk_size * action_dim)
        prediction = flat.view(batch_size, self.chunk_size, self.action_dim) # reshape to (B, chunk_size, action_dim)
        return torch.mean((prediction - action_chunk) ** 2) 
        
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        batch_size = state.shape[0]  
        flat = self.model(state) 
        prediction = flat.view(batch_size, self.chunk_size, self.action_dim) 
        return prediction

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
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.hidden_dims = hidden_dims

        self.flat_action_dim = self.action_dim * self.chunk_size
        self.in_dim = self.state_dim + self.flat_action_dim + 1

        layers: list[nn.Module] = []
        temp = self.in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(temp, h))
            layers.append(nn.ReLU())
            temp = h
        layers.append(nn.Linear(temp, self.flat_action_dim))
        self.net = nn.Sequential(*layers)




    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
         
        batch_size = state.shape[0]
        device = state.device
        A_t = action_chunk.view(batch_size, self.flat_action_dim)
        A_t0 = torch.rand(batch_size, self.flat_action_dim, device = device)
        tau = torch.rand(batch_size, 1, device=device)
        x_tau = (1.0 - tau) * A_t0 + tau * A_t
        v_target = A_t - A_t0
        input = torch.cat([state, x_tau, tau], dim=1)
        v_pred = self.net(input)
        return torch.mean((v_pred - v_target) ** 2)


    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        
        device = state.device
        batch_size = state.shape[0]
        A_t0 = torch.rand(batch_size, self.flat_action_dim, device=device)

        dt = 1.0 / float(num_steps)
        A_t_tau = A_t0

        for t in range(num_steps):
            tau = torch.full((batch_size, 1), t * dt, device=device)
            input = torch.cat([state, A_t_tau, tau], dim=1)
            v_pred = self.net(input)
            A_t_tau = A_t_tau + dt * v_pred
        return A_t_tau.view(batch_size, self.chunk_size, self.action_dim) 

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
