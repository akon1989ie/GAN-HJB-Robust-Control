# ================================================================
# Purpose: Robust GAN-HJB controller with stochastic dynamics,
# random initial points, deeper model, and multi-epoch training.
# ================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

# ---------------------------------------------------------------
# Activation dictionary
# ---------------------------------------------------------------
activation_dict = {
    "relu": lambda: nn.ReLU(inplace=False),
    "tanh": lambda: nn.Tanh(),
    "sigmoid": lambda: nn.Sigmoid(),
    "leaky_relu": lambda: nn.LeakyReLU(0.01, inplace=False),
    "elu": lambda: nn.ELU(alpha=1.0),
}


def get_ffnn(input_size, output_size, nn_desc, dropout_rate, bias):
    final_activ = None
    desc = nn_desc.copy()
    if isinstance(desc[-1], str):
        final_activ = desc[-1]
        desc = desc[:-1]
    layers = [nn.Linear(input_size, desc[0][0], bias=bias)]
    for i in range(len(desc)):
        act = desc[i][1]
        layers.append(activation_dict[act]())
        if dropout_rate > 0:
            layers.append(nn.Dropout(p=dropout_rate))
        if i < len(desc) - 1:
            layers.append(nn.Linear(desc[i][0], desc[i + 1][0], bias=bias))
    layers.append(nn.Linear(desc[-1][0], output_size, bias=bias))
    if final_activ:
        layers.append(activation_dict[final_activ]())
    return nn.Sequential(*layers)



# ---------------------------------------------------------------
# Running normalization
# ---------------------------------------------------------------
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
        var = self.M2 / max(self.count - 1, 1)
        std = torch.sqrt(var + 1e-6)
        return (x - self.mean) / std.clamp_min(1e-3)


# ---------------------------------------------------------------
# RNN Generator
# ---------------------------------------------------------------
class RNN(nn.Module):
    def __init__(self, hidden_desc, readout_desc, input_size, hidden_size,
                 output_size, dropout_rate=0.0, bias=True, control_scale=1.0):
        super().__init__()
        self.hidden = get_ffnn(input_size + hidden_size + 1, hidden_size, hidden_desc, dropout_rate, bias)
        self.readout = get_ffnn(hidden_size, output_size, readout_desc, dropout_rate, bias)
        self.hidden_size = hidden_size
        self.control_scale = control_scale
        self.h = None

    def reset_hidden(self, batch_size=1, device=None):
        device = device or torch.device("cpu")
        self.h = torch.zeros((batch_size, self.hidden_size), device=device)

    def forward(self, t, x):
        b = x.shape[0]
        if self.h is None or self.h.shape[0] != b:
            self.h = torch.zeros((b, self.hidden_size), device=x.device)
        if t.dim() == 1:
            t = t.view(1, 1).expand(b, -1)
        inp = torch.cat([x, self.h, t], dim=-1)
        self.h = self.hidden(inp)
        u = self.readout(self.h)
        return torch.tanh(u) * self.control_scale

# ---------------------------------------------------------------
# Discriminator
# ---------------------------------------------------------------
def snlinear(in_f, out_f):
    return nn.utils.spectral_norm(nn.Linear(in_f, out_f))


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            snlinear(input_dim + 1, hidden_size),
            nn.ReLU(),
            snlinear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.mu = snlinear(hidden_size, output_dim)
        self.sigma = snlinear(hidden_size, 1)

    def forward(self, t, x):
        if t.dim() == 1:
            t = t.view(1, 1).expand(x.shape[0], -1)
        h = self.net(torch.cat([x, t], dim=-1))
        mu = self.mu(h)
        sigma = F.softplus(self.sigma(h)) + 1e-6
        return mu, sigma


# ---------------------------------------------------------------
# MinMAx Combined Model (Corrected)
# ---------------------------------------------------------------
class MinMAx(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hidden_desc, readout_desc,
                 dropout_rate=0.0, bias=True, output_size_disc=None, control_scale=1.0):
        super().__init__()
        self.rnn = RNN(hidden_desc, readout_desc, input_size, hidden_size, output_size,
                       dropout_rate, bias, control_scale)
        disc_out = 1 if output_size_disc is None else output_size_disc
        self.disc = Discriminator(input_size, hidden_size, disc_out)

    def forward(self, t, inputs):
        return self.rnn(t, inputs), self.disc(t, inputs)

    


# ---------------------------------------------------------------
# HJB + Loss
# ---------------------------------------------------------------
class HJBModel:
    def __init__(self, X, mu, u, sigma, Q, R, x_target, device):
        self.X, self.mu, self.u = X.view(-1, 1), mu.view(-1, 1), u.view(-1, 1)
        self.sigma = sigma.view(-1)[0]
        self.Q, self.R = Q, R
        self.x_target = x_target.view(-1, 1)
        self.device = device
        omega = 0.6
        self.f = torch.tensor([[0, 0, 1, 0],
                               [0, 0, 0, 1],
                               [0, omega, 0, 0],
                               [-omega, 0, 0, 0]], device=device)
        self.G = torch.tensor([[0.25, 0.0],
                               [0.0, 0.25],
                               [1.0, 0.0],
                               [0.0, 1.0]], device=device)
        self.COV = torch.tensor([[0, 0],
                                 [0, 0],
                                 [0.5, 0],
                                 [0, 0.5]], device=device)

    def HJB_exp(self):
        xr = self.X - self.x_target
        V = xr.T @ self.Q @ xr
        grad1 = 2 * (self.X - self.x_target).T @ self.Q
        grad2 = 2 * self.Q
        trace_term = 0.5 * (self.sigma ** 2) * torch.trace(grad2 @ (self.COV @ self.COV.T))
        dynamics = self.f @ self.X + self.G @ self.u + self.COV @ self.mu
        return (grad1 @ dynamics + V + 0.5 * self.u.T @ self.R @ self.u + trace_term).squeeze()
        


    


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



