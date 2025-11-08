# ================================================================
# File: gan_hjb_curved_targeted_optimized_obstacle_v5.py
# Purpose: Robust GAN-HJB controller with obstacle avoidance,
#          random starts, multi-iteration per sample,
#          and stronger / smoother obstacle bypass
# ================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

# ----------------------------------------------------------------------
# Activation functions
# ----------------------------------------------------------------------
activation_dict = {
    "relu": lambda: nn.ReLU(inplace=False),
    "tanh": lambda: nn.Tanh(),
    "sigmoid": lambda: nn.Sigmoid(),
    "leaky_relu": lambda: nn.LeakyReLU(negative_slope=0.01, inplace=False),
    "elu": lambda: nn.ELU(alpha=1.0),
}

def get_ffnn(input_size, output_size, nn_desc, dropout_rate, bias):
    desc = nn_desc.copy()
    final_activ = None
    if isinstance(desc[-1], str):
        final_activ = desc[-1]
        desc = desc[:-1]
    layers = [nn.Linear(input_size, desc[0][0], bias=bias)]
    for i in range(len(desc)):
        layers.append(activation_dict[desc[i][1]]())
        if dropout_rate > 0:
            layers.append(nn.Dropout(p=dropout_rate))
        if i < len(desc) - 1:
            layers.append(nn.Linear(desc[i][0], desc[i + 1][0], bias=bias))
    layers.append(nn.Linear(desc[-1][0], output_size, bias=bias))
    if final_activ:
        layers.append(activation_dict[final_activ]())
    return nn.Sequential(*layers)

# ----------------------------------------------------------------------
# Running normalization
# ----------------------------------------------------------------------
class RunningNorm:
    def __init__(self, dim, device):
        self.count = 1e-6
        self.mean = torch.zeros(dim, device=device)
        self.M2 = torch.zeros(dim, device=device)

    @torch.no_grad()
    def update(self, x):
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def normalize(self, x):
        var = self.M2 / max(self.count - 1, 1.0)
        std = torch.sqrt(var + 1e-6)
        return (x - self.mean) / std.clamp_min(1e-3)

# ----------------------------------------------------------------------
# RNN (Generator)
# ----------------------------------------------------------------------
class RNN(nn.Module):
    def __init__(self, hidden_desc, readout_desc, input_size, hidden_size, output_size,
                 dropout_rate=0.0, bias=True, control_scale=1.0):
        super().__init__()
        self.hidden = get_ffnn(input_size + hidden_size + 1, hidden_size, hidden_desc, dropout_rate, bias)
        self.readout = get_ffnn(hidden_size, output_size, readout_desc, dropout_rate, bias)
        self.hidden_size = hidden_size
        self.control_scale = control_scale
        self.h = None

    def reset_hidden(self, batch_size=1, device=None):
        self.h = torch.zeros((batch_size, self.hidden_size), device=device)

    def forward(self, t, x):
        b = x.shape[0]
        if self.h is None or self.h.shape[0] != b:
            self.h = torch.zeros((b, self.hidden_size), device=x.device)
        if t.dim() == 1 and t.numel() == 1:
            t = t.view(1, 1).expand(b, -1)
        inp = torch.cat([x, self.h, t], dim=-1)
        self.h = self.hidden(inp)
        u = torch.tanh(self.readout(self.h)) * self.control_scale
        return u

# ----------------------------------------------------------------------
# Discriminator
# ----------------------------------------------------------------------
def snlinear(i, o):
    return nn.utils.spectral_norm(nn.Linear(i, o))

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            snlinear(input_dim + 1, hidden_size), nn.ReLU(),
            snlinear(hidden_size, hidden_size), nn.ReLU()
        )
        self.mu = snlinear(hidden_size, output_dim)
        self.sigma = snlinear(hidden_size, 1)

    def forward(self, t, x):
        if t.dim() == 1 and t.numel() == 1:
            t = t.view(1, 1).expand(x.shape[0], -1)
        h = self.net(torch.cat([x, t], dim=-1))
        return self.mu(h), F.softplus(self.sigma(h)) + 1e-6

# ----------------------------------------------------------------------
# Combined model
# ----------------------------------------------------------------------
class MinMAx(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hidden_desc, readout_desc,
                 dropout_rate=0.0, bias=True, control_scale=1.0):
        super().__init__()
        self.rnn = RNN(
            hidden_desc, readout_desc,
            input_size, hidden_size, output_size,
            dropout_rate, bias, control_scale
        )
        self.disc = Discriminator(input_size, hidden_size, output_size)

# ----------------------------------------------------------------------
# HJB Loss
# ----------------------------------------------------------------------
class HJBModel:
    def __init__(self, X, mu, u, sigma, Q, R, x_target, device):
        self.X = X.view(-1, 1)
        self.mu = mu.view(-1, 1)
        self.u = u.view(-1, 1)
        self.sigma = sigma.view(-1)[0]
        self.Q = Q
        self.R = R
        self.x_target = x_target.view(-1, 1)
        self.device = device

        omega = 0.6
        self.f = torch.tensor(
            [[0,     0,     1, 0],
             [0,     0,     0, 1],
             [0,  omega,    0, 0],
             [-omega, 0,    0, 0]],
            dtype=torch.float32, device=device
        )
        self.G = torch.tensor(
            [[0.3, 0],
             [0,   0.25],
             [1,   0],
             [0,   1]],
            dtype=torch.float32, device=device
        )
        self.COV = torch.tensor(
            [[0,   0],
             [0,   0],
             [0.5, 0],
             [0,   0.5]],
            dtype=torch.float32, device=device
        )

    def HJB_exp(self):
        xr = self.X - self.x_target
        V = xr.T @ self.Q @ xr
        grad1 = 2 * (self.X - self.x_target).T @ self.Q
        grad2 = 2 * self.Q
        trace_term = 0.5 * (self.sigma ** 2) * torch.trace(grad2 @ (self.COV @ self.COV.T))
        dyn = self.f @ self.X + self.G @ self.u + self.COV @ self.mu
        return (grad1 @ dyn + V + 0.5 * self.u.T @ self.R @ self.u + trace_term).squeeze()

class HJBLoss(nn.Module):
    def __init__(self, Q, R, x_target, device):
        super().__init__()
        self.Q = Q
        self.R = R
        self.x_target = x_target
        self.device = device

    def forward(self, X, mu, sigma, u):
        loss = 0.0
        for i in range(X.shape[0]):
            hjb = HJBModel(X[i], mu[i], u[i], sigma[i],
                           self.Q, self.R, self.x_target, self.device)
            loss += hjb.HJB_exp()
        return loss / X.shape[0]


class Penalty:
    def __init__(self, lambda1: torch.Tensor, lambda2: torch.Tensor, terms: str):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.terms = terms

    def get_penalty_function(self, desc: str, reference: torch.Tensor):
        p = desc.split('-')[0]
        assert p == "fro", "Only Frobenius norm penalty supported for now."

        if self.terms == "drift":
            # Use pseudo-inverse for stability
            y_inv = torch.linalg.pinv(reference.T @ reference)

            def penalty(mu: torch.Tensor) -> torch.Tensor:
                """
                Args:
                    mu: Tensor of shape (batch, d) or (batch, d, d)
                """
                if mu.dim() == 2:  # lift to (batch, d, d)
                    mu = mu.unsqueeze(2) @ mu.unsqueeze(1)

                M = (mu @ mu.transpose(1, 2)) - y_inv
                if "squared" in desc:
                    penalties = torch.sum(M ** 2, dim=(1, 2))
                else:
                    penalties = torch.linalg.norm(M, ord=p, dim=(1, 2))
                return self.lambda1 * penalties.mean()
            return penalty

        else:  # sigma penalty
            def penalty(sigma: torch.Tensor = None) -> torch.Tensor:
                """
                Args:
                    sigma: Tensor of shape (batch, 1)
                """
                sigma = sigma.unsqueeze(2) @ sigma.unsqueeze(1)  # (batch,1,1)
                penalties = torch.linalg.norm(sigma - reference, ord="fro", dim=(1, 2))
                return self.lambda2 * penalties.mean()
            return penalty



