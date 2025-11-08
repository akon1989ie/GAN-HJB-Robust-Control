import os
from absl import flags
import ml_collections
import wandb
import configs
from Train import HJBLoss, TrainEpochHJB, train_hjb
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
