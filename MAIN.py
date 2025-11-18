import os
from absl import flags
import ml_collections
import wandb
import configs
from Train import HJBLoss, TrainEpochHJB, train_hjb, plot_trajectory
import torch
from run import *
torch.autograd.set_detect_anomaly(True)
if __name__ == "__main__":
    config = {
        "Iter": 100,   # can increase to 150–200 for even better paths
        "Inner": 5,
        "dt": 0.02,
        "T": 7.0
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Q = torch.diag(torch.tensor([80.0, 80.0, 2.0, 2.0], device=device))
    R = torch.eye(2, device=device) * 0.01
    x_target = torch.zeros((4,), device=device)

    hidden_desc = [(128, "relu"), (96, "tanh")]
    readout_desc = [(64, "relu"), "tanh"]

    model = MinMAx(
        input_size=8,
        hidden_size=128,
        output_size=2,
        hidden_desc=hidden_desc,
        readout_desc=readout_desc,
        dropout_rate=0.1,
        bias=True,
        control_scale=1.8
    ).to(device)

    opt_rnn = torch.optim.AdamW(model.rnn.parameters(), lr=5e-4, weight_decay=1e-4)
    opt_disc = torch.optim.AdamW(model.disc.parameters(), lr=3e-4, weight_decay=1e-4)

    trainer = TrainEpochHJB(model, opt_rnn, opt_disc, device, Q, R, x_target)

    base_start = torch.tensor([4.0, -1.5, 0.0, 0.0], device=device)

    for ep in range(1, config["Iter"] + 1):
        perturb = torch.randn(4, device=device) * torch.tensor([0.25, 0.25, 0.05, 0.05], device=device)
        x0 = base_start + perturb
        g, d = trainer.run(x0, config["dt"], config["T"], config["Inner"])
        print(f"[{ep:03d}] Gen={g:.3f} | Disc={d:.3f} | Start={x0[:2].tolist()}")

    print("Evaluating multiple trajectories...")
    for i in range(3):
        perturb = torch.randn(4, device=device) * torch.tensor([0.25, 0.25, 0.05, 0.05], device=device)
        x0 = base_start + perturb
        x_hist, _, _, _ = trainer.simulate_trajectory(x0, config["dt"], config["T"], build_graph=False)
        plot_trajectory(
            x_hist,
            trainer.obstacle_center.detach().cpu().numpy(),
            trainer.obstacle_size,
            title=f"Trajectory from random start {i+1}"
        )

# ================================================================
# File: gan_hjb_curved_targeted_optimized_obstacle_v8_1.py
# Purpose: Robust GAN-HJB controller with obstacle avoidance,
#          random starts, multi-iteration per sample,
#          and hinge-style hard-barrier obstacle penalty
#          (w_obstacle kept fixed at 1200.0)
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
        self.count += 1.0
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def normalize(self, x):
        var = self.M2 / max(self.count - 1.0, 1.0)
        std = torch.sqrt(var + 1e-6)
        return (x - self.mean) / std.clamp_min(1e-3)


# ----------------------------------------------------------------------
# RNN (Generator)
# ----------------------------------------------------------------------
class RNN(nn.Module):
    def __init__(
        self,
        hidden_desc,
        readout_desc,
        input_size,
        hidden_size,
        output_size,
        dropout_rate=0.0,
        bias=True,
        control_scale=1.0,
    ):
        super().__init__()
        self.hidden = get_ffnn(
            input_size + hidden_size + 1,
            hidden_size,
            hidden_desc,
            dropout_rate,
            bias,
        )
        self.readout = get_ffnn(
            hidden_size,
            output_size,
            readout_desc,
            dropout_rate,
            bias,
        )
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
            snlinear(input_dim + 1, hidden_size),
            nn.ReLU(),
            snlinear(hidden_size, hidden_size),
            nn.ReLU(),
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
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        hidden_desc,
        readout_desc,
        dropout_rate=0.0,
        bias=True,
        control_scale=1.0,
    ):
        super().__init__()
        self.rnn = RNN(
            hidden_desc,
            readout_desc,
            input_size,
            hidden_size,
            output_size,
            dropout_rate,
            bias,
            control_scale,
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
            [
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [0, omega, 0, 0],
                [-omega, 0, 0, 0],
            ],
            dtype=torch.float32,
            device=device,
        )
        self.G = torch.tensor(
            [
                [0.25, 0.0],
                [0.0, 0.25],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )
        self.COV = torch.tensor(
            [
                [0, 0],
                [0, 0],
                [0.5, 0],
                [0, 0.5],
            ],
            dtype=torch.float32,
            device=device,
        )

    def HJB_exp(self):
        xr = self.X - self.x_target
        V = xr.T @ self.Q @ xr
        grad1 = 2 * (self.X - self.x_target).T @ self.Q
        grad2 = 2 * self.Q
        trace_term = 0.5 * (self.sigma**2) * torch.trace(
            grad2 @ (self.COV @ self.COV.T)
        )
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
            hjb = HJBModel(
                X[i], mu[i], u[i], sigma[i], self.Q, self.R, self.x_target, self.device
            )
            loss += hjb.HJB_exp()
        return loss / X.shape[0]


# ----------------------------------------------------------------------
# Trainer
# ----------------------------------------------------------------------
class TrainEpochHJB:
    def __init__(self, model, opt_rnn, opt_disc, device, Q, R, x_target):
        self.model = model
        self.opt_rnn = opt_rnn
        self.opt_disc = opt_disc
        self.device = device
        self.Q = Q
        self.R = R
        self.x_target = x_target
        self.hjb_loss = HJBLoss(Q, R, x_target, device)
        self.normer = RunningNorm(4, device)

        # Dynamics for simulation
        self.f = torch.tensor(
            [
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [0, 0.0, 0, 0],
                [0.0, 0, 0, 0],
            ],
            dtype=torch.float32,
            device=device,
        )
        self.G = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )
        self.COV = torch.tensor(
            [
                [0, 0],
                [0, 0],
                [0.5, 0],
                [0, 0.5],
            ],
            dtype=torch.float32,
            device=device,
        )

        # Obstacle parameters
        self.obstacle_center = torch.tensor([-2.2, 1.2], device=device)
        self.obstacle_size = 2.0  # side length of square

        # Outside wall: Gaussian
        self.gauss_sigma = 0.5

        # Hinge barrier parameters
        self.inner_margin = 0.02  # activation margin near the wall

        # Loss weights (w_obstacle fixed as requested)
        self.w_hjb = 0.5
        self.w_terminal = 100.0
        self.w_obstacle = 1200.0   # <-- fixed
        self.w_smooth = 2e-3

    # Noise in dynamics
    def get_dynamic_X(self, mu, sigma, dt=0.02):
        dw = torch.randn_like(mu, device=self.device)
        return mu * dt + sigma * dw * torch.sqrt(torch.tensor(dt, device=self.device))

    # ---------------- Hinge-style hard-barrier obstacle penalty ----------------
    def obstacle_penalty(self, x):
        """
        Hard-ish barrier:

        - Outside the square: smooth Gaussian near boundary, ~0 far away.
        - Slightly inside or right at the wall: hinge^2 penalty (strong but controlled).
        """
        pos = x[:2]
        cx, cy = self.obstacle_center
        half = self.obstacle_size / 2.0

        dx = torch.abs(pos[0] - cx) - half
        dy = torch.abs(pos[1] - cy) - half

        # outside distance (Euclidean to nearest point on square)
        outside_dx = torch.clamp(dx, min=0.0)
        outside_dy = torch.clamp(dy, min=0.0)
        outside_dist = torch.sqrt(outside_dx**2 + outside_dy**2)

        # inside distance (negative, using max(dx, dy))
        inside_dist = torch.max(dx, dy)

        # signed distance: >0 outside, <=0 inside
        signed_dist = torch.where(
            (dx > 0) | (dy > 0),
            outside_dist,
            inside_dist,
        )

        # Outside: Gaussian wall (strong only near boundary)
        outside_penalty = torch.exp(
            -(signed_dist.clamp_min(0.0) / self.gauss_sigma) ** 2
        )

        # Inside / slightly near boundary: hinge^2
        # hinge = max(0, -signed_dist + margin)
        margin = self.inner_margin
        hinge = torch.relu(-signed_dist + margin)
        inside_penalty = 800.0 * (hinge**2)  # scale chosen to match your regime

        penalty = torch.where(
            signed_dist >= 0.0,
            outside_penalty,
            inside_penalty,
        )
        return penalty

    # --------------------------------------------------------------------------
    def simulate_trajectory(self, x0, dt, T, build_graph=True):
        n_steps = int(T / dt)
        x = x0.clone().to(self.device)
        self.model.rnn.reset_hidden(1, self.device)

        xs, mus, sigmas, us = [], [], [], []
        ctx = torch.enable_grad() if build_graph else torch.no_grad()
        with ctx:
            for t in torch.linspace(0.0, T, n_steps, device=self.device):
                tt = torch.tensor([[t.item()]], device=self.device)

                # normalize state
                self.normer.update(x.detach())
                x_in = self.normer.normalize(x).view(1, -1)
                inp = torch.cat([x_in, self.x_target.view(1, -1)], dim=-1)

                # adversarial drift & control
                mu, sigma = self.model.disc(tt, inp)
                u = self.model.rnn(tt, inp)

                eps = self.get_dynamic_X(mu, sigma, dt).detach()
                dx = (
                    (self.f @ x.view(-1, 1)).view(-1)
                    + (self.G @ u.view(-1, 1)).view(-1)
                    + (self.COV @ eps.view(-1, 1)).view(-1)
                )
                x = x + dt * dx

                xs.append(x.clone())
                mus.append(mu.view(-1))
                sigmas.append(sigma.view(-1))
                us.append(u.view(-1))

        return (
            torch.stack(xs),
            torch.stack(mus),
            torch.stack(sigmas),
            torch.stack(us),
        )

    # --------------------------------------------------------------------------
    def run(self, x0, dt, T, inner_iter=5):
        g_total, d_total = 0.0, 0.0

        for _ in range(inner_iter):
            # ------------------------------- Generator update -------------------
            x_hist, mu_hist, sigma_hist, u_hist = self.simulate_trajectory(x0, dt, T)
            hjb = self.hjb_loss(x_hist, mu_hist, sigma_hist, u_hist)

            # terminal distance to target (position only)
            terminal_dist = x_hist[-1, 0] ** 2 + x_hist[-1, 1] ** 2

            # control smoothness
            smooth = (u_hist[1:] - u_hist[:-1]).pow(2).mean()

            # obstacle cost along the full path
            obstacle_term = torch.stack(
                [self.obstacle_penalty(x_hist[i]) for i in range(len(x_hist))]
            ).mean()

            loss_g = (
                self.w_hjb * hjb
                + self.w_terminal * terminal_dist
                + self.w_smooth * smooth
                + self.w_obstacle * obstacle_term
            )

            self.opt_rnn.zero_grad()
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(self.model.rnn.parameters(), 5.0)
            self.opt_rnn.step()
            g_total += float(loss_g.detach())

            # ------------------------------- Discriminator update --------------
            x_hist_d, mu_hist_d, sigma_hist_d, u_hist_d = self.simulate_trajectory(
                x0, dt, T
            )
            loss_d = -self.hjb_loss(x_hist_d, mu_hist_d, sigma_hist_d, u_hist_d)
            self.opt_disc.zero_grad()
            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(self.model.disc.parameters(), 5.0)
            self.opt_disc.step()
            d_total += float(loss_d.detach())

        return g_total / inner_iter, d_total / inner_iter


# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------
def plot_trajectory(
    x_hist, center, size, title="Trajectory with Obstacle (Hard-Barrier v8.1)"
):
    x = x_hist.detach().cpu().numpy()
    cx, cy = center
    half = size / 2
    square = np.array(
        [
            [cx - half, cy - half],
            [cx + half, cy - half],
            [cx + half, cy + half],
            [cx - half, cy + half],
            [cx - half, cy - half],
        ]
    )

    plt.figure(figsize=(6, 6))
    t_idx = np.linspace(0, 1, len(x))
    plt.scatter(x[:, 0], x[:, 1], c=t_idx, s=10, cmap="viridis", label="Trajectory")
    plt.plot(x[:, 0], x[:, 1], linewidth=1.5, alpha=0.7)

    plt.scatter(x[0, 0], x[0, 1], c="g", s=80, edgecolor="k", label="Start")
    plt.scatter(0.0, 0.0, c="k", s=100, marker="*", label="Target")
    plt.plot(square[:, 0], square[:, 1], "r--", lw=2, label="Obstacle")

    plt.colorbar(label="Time along trajectory")
    plt.legend()
    plt.axis("equal")
    plt.grid(True, linestyle=":")
    plt.xlabel("x position")
    plt.ylabel("y position")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    config = {
        "Iter": 140,   # total outer epochs
        "Inner": 3,    # inner iterations per epoch
        "dt": 0.02,
        "T": 8.0,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Q = torch.diag(torch.tensor([80.0, 80.0, 2.0, 2.0], device=device))
    R = torch.eye(2, device=device) * 0.01
    x_target = torch.zeros((4,), device=device)

    hidden_desc = [(128, "relu"), (96, "tanh")]
    readout_desc = [(64, "relu"), "tanh"]

    model = MinMAx(
        input_size=8,
        hidden_size=128,
        output_size=2,
        hidden_desc=hidden_desc,
        readout_desc=readout_desc,
        dropout_rate=0.1,
        bias=True,
        control_scale=1.5,   # slightly smaller than v7 for smoother paths
    ).to(device)

    # Slightly smaller LR for stability with strong obstacle penalties
    opt_rnn = torch.optim.AdamW(model.rnn.parameters(), lr=5e-4, weight_decay=1e-5)
    opt_disc = torch.optim.AdamW(model.disc.parameters(), lr=1.2e-4, weight_decay=5e-5)

    trainer = TrainEpochHJB(model, opt_rnn, opt_disc, device, Q, R, x_target)

    base_start = torch.tensor([-3, 2.5, 0.0, 0.0], device=device)

    for ep in range(1, config["Iter"] + 1):
        perturb = torch.randn(4, device=device) * torch.tensor(
            [0.15, 0.15, 0.05, 0.05], device=device
        )
        x0 = base_start + perturb

        g, d = trainer.run(x0, config["dt"], config["T"], config["Inner"])
        print(
            f"[{ep:03d}] Gen={g:.3f} | Disc={d:.3f} "
            f"| w_obs={trainer.w_obstacle:.1f} | Start={x0[:2].tolist()}"
        )

    print("Evaluating multiple trajectories (Hard-Barrier v8.1)...")
    for i in range(3):
        perturb = torch.randn(4, device=device) * torch.tensor(
            [0.15, 0.15, 0.05, 0.05], device=device
        )
        x0 = base_start + perturb
        x_hist, _, _, _ = trainer.simulate_trajectory(
            x0, config["dt"], config["T"], build_graph=False
        )
        plot_trajectory(
            x_hist,
            trainer.obstacle_center.detach().cpu().numpy(),
            trainer.obstacle_size,
            title=f"Trajectory from random start {i+1} (Hard-Barrier v8.1)",
        )
