import torch
import torch.nn as nn
from typing import Optional
import os
from run import HJBModel, MinMAx, Penalty, RunningNorm # keep your run module imports
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(0)

class HJBLoss(nn.Module):
    def __init__(self, state_dim=4, control_dim=2, device='cpu'):
        super().__init__()
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.device = device

    def forward(self, X, mu, u):
        """
        X: Tensor (time_steps, state_dim)  OR (batch_size, state_dim)
        mu: Tensor (time_steps, mu_dim)
        u: Tensor (time_steps, control_dim)
        Returns scalar tensor (keeps grad)
        """
        # Ensure we accumulate with a tensor that participates in autograd
        loss = torch.zeros(1, dtype=torch.float32, device=self.device)
        # assume first dim is time/batch
        for i in range(X.shape[0]):
            hjb = HJBModel(
                X=X[i].view(self.state_dim, 1),
                mu=mu[i].view(mu.shape[1], 1),
                u=u[i].view(self.control_dim, 1),
                state_dim=self.state_dim,
                control_dim=self.control_dim,
                device=self.device
            )
            # HJB_exp should return a tensor (scalar)
            loss = loss + hjb.HJB_exp()
        loss = loss / X.shape[0]
        return loss.squeeze()  # scalar tensor


torch.manual_seed(0)

class HJBLoss(nn.Module):
    def __init__(self, Q, R, x_target, device):
        super().__init__()
        self.Q, self.R, self.x_target, self.device = Q, R, x_target, device

    def forward(self, X, mu, sigma, u):
        loss = 0
        for i in range(X.shape[0]):
            hjb = HJBModel(X[i], mu[i], u[i], sigma[i], self.Q, self.R, self.x_target, self.device)
            loss += hjb.HJB_exp()
        return loss / X.shape[0]





# ---------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------
class TrainEpochHJB:
    def __init__(self, model, opt_rnn, opt_disc, device, Q, R, x_target):
        self.model, self.opt_rnn, self.opt_disc = model, opt_rnn, opt_disc
        self.device, self.Q, self.R, self.x_target = device, Q, R, x_target
        self.hjb_loss = HJBLoss(Q, R, x_target, device)
        self.f = torch.tensor([[0, 0, 1, 0],
                               [0, 0, 0, 1],
                               [0, 0.3, 0, 0],
                               [-0.3, 0, 0, 0]], device=device)
        self.G = torch.tensor([[0.4, 0], [0, 0.25], [1, 0], [0, 1]], device=device)
        self.COV = torch.tensor([[0, 0], [0, 0], [0.5, 0], [0, 0.5]], device=device)
        self.normer = RunningNorm(4, device)

    def get_dynamic_X(self, mu, sigma, dt):
        dw = torch.randn_like(mu, device=self.device)
        return mu * dt + sigma * dw * torch.sqrt(torch.tensor(dt, device=self.device))

    def simulate_trajectory(self, x0, dt, T, build_graph=True):
        n_steps = int(T / dt)
        x = x0.clone().to(self.device)
        self.model.rnn.reset_hidden(device=self.device)
        xs, mus, sigmas, us = [], [], [], []
        ctx = torch.enable_grad() if build_graph else torch.no_grad()
        with ctx:
            for t in torch.linspace(0, T, n_steps):
                tt = torch.tensor([[t.item()]], device=self.device)
                self.normer.update(x.detach())
                x_in = self.normer.normalize(x).view(1, -1)
                inp = torch.cat([x_in, self.x_target.view(1, -1)], dim=-1)
                mu, sigma = self.model.disc(tt, inp)
                u = self.model.rnn(tt, inp)
                eps = self.get_dynamic_X(mu, sigma, dt).detach()
                dx = (self.f @ x.view(-1, 1)).view(-1) + (self.G @ u.view(-1, 1)).view(-1) + (self.COV @ eps.view(-1, 1)).view(-1)
                x = x + dt * dx
                xs.append(x.clone())
                mus.append(mu.view(-1))
                sigmas.append(sigma.view(-1))
                us.append(u.view(-1))
        return torch.stack(xs), torch.stack(mus), torch.stack(sigmas), torch.stack(us)

    def run(self, x0, dt, T):
        x_hist, mu_hist, sigma_hist, u_hist = self.simulate_trajectory(x0, dt, T, True)
        hjb = self.hjb_loss(x_hist, mu_hist, sigma_hist, u_hist)
        term = 200 * ((x_hist[-1, 0]) ** 2 + (x_hist[-1, 1]) ** 2)
        smooth = 1e-5 * (u_hist[1:] - u_hist[:-1]).pow(2).mean()
        loss_g = hjb + term + smooth
        self.opt_rnn.zero_grad()
        loss_g.backward()
        torch.nn.utils.clip_grad_norm_(self.model.rnn.parameters(), 5)
        self.opt_rnn.step()

        x_hist, mu_hist, sigma_hist, u_hist = self.simulate_trajectory(x0, dt, T, True)
        hjb_d = self.hjb_loss(x_hist, mu_hist, sigma_hist, u_hist)
        term_d = 200 * ((x_hist[-1, 0]) ** 2 + (x_hist[-1, 1]) ** 2)
        smooth_d = 1e-5 * (u_hist[1:] - u_hist[:-1]).pow(2).mean()
        loss_d = -(hjb_d + term_d + smooth_d)
        self.opt_disc.zero_grad()
        loss_d.backward()
        torch.nn.utils.clip_grad_norm_(self.model.disc.parameters(), 5)
        self.opt_disc.step()

        return float(loss_g.detach()), float(loss_d.detach())


# ---------------------------------------------------------------
# Multi-epoch Random Training Loop
# ---------------------------------------------------------------
def train_hjb(model, opt_rnn, opt_disc, device, Q, R, x_target,
              num_epochs=150, inner_epochs=6, T=7.0, dt=0.02):
    trainer = TrainEpochHJB(model, opt_rnn, opt_disc, device, Q, R, x_target)
    sch_rnn = torch.optim.lr_scheduler.CosineAnnealingLR(opt_rnn, T_max=num_epochs)
    sch_disc = torch.optim.lr_scheduler.CosineAnnealingLR(opt_disc, T_max=num_epochs)

    for ep in range(1, num_epochs + 1):
        x0_np = np.random.uniform([3, -2, 0, 0], [5, -3, 0, 0])
        x0 = torch.tensor(x0_np, dtype=torch.float32, device=device)
        for inner in range(inner_epochs):
            g, d = trainer.run(x0, dt, T)
            print(f"[Epoch {ep}/{num_epochs}] (Inner {inner + 1}/{inner_epochs}) "
                  f"x0={x0_np.round(2)} | Gen={g:.4f} | Disc={d:.4f}")
        sch_rnn.step()
        sch_disc.step()
    return trainer

# ---------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------
def plot_trajectory(x_hist, title="Trajectory"):
    x = x_hist.cpu().numpy()
    plt.figure(figsize=(6, 6))
    plt.plot(x[:, 0], x[:, 1], "-o", ms=3)
    plt.scatter(x[0, 0], x[0, 1], c="g", s=80, label="Start")
    plt.scatter(0, 0, c="k", s=100, marker="*", label="Target")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.title(title)
    plt.show()


def plot_controls(u_hist, dt):
    u = u_hist.cpu().numpy()
    t = np.arange(len(u)) * dt
    plt.figure(figsize=(8, 4))
    plt.plot(t, u[:, 0], label="u1")
    plt.plot(t, u[:, 1], label="u2")
    plt.legend()
    plt.grid(True)
    plt.title("Control Inputs vs Time")
    plt.show()