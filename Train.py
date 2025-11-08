import torch
import torch.nn as nn
from typing import Optional
import os
from run import HJBModel, MinMAx, Penalty, RunningNorm # keep your run module imports
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(0)

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

        # Dynamics (can be slightly different from HJBModel if desired)
        self.f = torch.tensor(
            [[0,   0,   1, 0],
             [0,   0,   0, 1],
             [0, 0.3,   0, 0],
             [-0.3, 0,  0, 0]],
            dtype=torch.float32, device=device
        )
        self.G = torch.tensor(
            [[0.4, 0.0],
             [0.0, 0.25],
             [1.0, 0.0],
             [0.0, 1.0]],
            dtype=torch.float32, device=device
        )
        self.COV = torch.tensor(
            [[0,   0],
             [0,   0],
             [0.5, 0],
             [0,   0.5]],
            dtype=torch.float32, device=device
        )

        # Obstacle parameters
        self.obstacle_center = torch.tensor([2.0, -1.0], device=device)
        self.obstacle_size = 0.8
        self.margin = 0.9   # Increased margin around obstacle
        self.safe_radius = 0.5 * self.obstacle_size + self.margin

        # Loss weights: tune trade-off between target vs obstacle
        self.w_hjb = 0.5
        self.w_terminal = 50.0
        self.w_obstacle = 1200.0
        self.w_smooth = 5e-5

    def get_dynamic_X(self, mu, sigma, dt=0.02):
        dw = torch.randn_like(mu, device=self.device)
        return mu * dt + sigma * dw * torch.sqrt(torch.tensor(dt, device=self.device))

    def obstacle_penalty(self, x):
        """
        Smooth 'safety bubble' around obstacle:
        penalty ~ softplus(r_safe - dist)^2, large when inside bubble
        """
        pos = x[:2]
        dist = torch.norm(pos - self.obstacle_center)
        d = self.safe_radius - dist
        return F.softplus(d)**2

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

        return torch.stack(xs), torch.stack(mus), torch.stack(sigmas), torch.stack(us)

    def run(self, x0, dt, T, inner_iter=5):
        g_total, d_total = 0.0, 0.0

        for _ in range(inner_iter):
            # -------------------------------
            # Generator (controller) update
            # -------------------------------
            x_hist, mu_hist, sigma_hist, u_hist = self.simulate_trajectory(x0, dt, T)
            hjb = self.hjb_loss(x_hist, mu_hist, sigma_hist, u_hist)

            # terminal distance to target
            terminal_dist = (x_hist[-1, 0]**2 + x_hist[-1, 1]**2)

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

            # -------------------------------
            # Discriminator update
            # -------------------------------
            x_hist_d, mu_hist_d, sigma_hist_d, u_hist_d = self.simulate_trajectory(x0, dt, T)
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
def plot_trajectory(x_hist, center, size, title="Trajectory with Obstacle"):
    x = x_hist.detach().cpu().numpy()
    cx, cy = center
    half = size / 2
    square = np.array([
        [cx - half, cy - half],
        [cx + half, cy - half],
        [cx + half, cy + half],
        [cx - half, cy + half],
        [cx - half, cy - half],
    ])

    plt.figure(figsize=(6, 6))
    plt.plot(x[:, 0], x[:, 1], "-o", ms=3, label="Trajectory")
    plt.scatter(x[0, 0], x[0, 1], c="g", s=80, label="Start")
    plt.scatter(0.0, 0.0, c="k", s=100, marker="*", label="Target")
    plt.plot(square[:, 0], square[:, 1], "r--", lw=2, label="Obstacle")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.title(title)
    plt.show()
