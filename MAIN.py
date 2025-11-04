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
        "state_dim": 4,
        "act_dim": 2,
        "Dropout": 0.15,
        "lr_g": 5e-4,
        "lr_d": 3e-4,
        "weight_decay": 5e-4,
        "Iter": 50,
        "Inner": 40,
        "T": 7.0,
        "dt": 0.02,
        "control_scale": 1.5,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Q = torch.diag(torch.tensor([80.0, 80.0, 2.0, 2.0], dtype=torch.float32))
    R = torch.eye(2) * 0.01
    x_target = torch.zeros((4,), device=device)

    hidden_desc = [(128, "relu"), (96, "tanh"), (64, "relu"), (48, "tanh")]
    readout_desc = [(64, "relu"), (32, "tanh"), "tanh"]

    model = MinMAx(input_size=8, hidden_size=128, output_size=2,
                   hidden_desc=hidden_desc, readout_desc=readout_desc,
                   dropout_rate=config["Dropout"], bias=True,
                   output_size_disc=2, control_scale=config["control_scale"]).to(device)

    opt_rnn = torch.optim.AdamW(model.rnn.parameters(), lr=config["lr_g"], weight_decay=config["weight_decay"])
    opt_disc = torch.optim.AdamW(model.disc.parameters(), lr=config["lr_d"], weight_decay=config["weight_decay"])

    print("Training GAN-HJB with random initial points + deeper network...")
    trainer = train_hjb(model, opt_rnn, opt_disc, device, Q, R, x_target,
                        num_epochs=config["Iter"], inner_epochs=config["Inner"],
                        T=config["T"], dt=config["dt"])

    print("Training complete. Simulating final rollout from (4, -1.5, 0, 0)...")
    model.eval()
    with torch.no_grad():
        x0_eval = torch.tensor([4, -1.5, 0, 0], dtype=torch.float32, device=device)
        x_hist, mu_hist, sigma_hist, u_hist = trainer.simulate_trajectory(x0_eval, config["dt"], config["T"], False)

    print(f"Final position: ({x_hist[-1, 0]:.3f}, {x_hist[-1, 1]:.3f})")
    plot_trajectory(x_hist, "GAN-HJB (Robust, Random Init, Deep Model)")
    plot_controls(u_hist, config["dt"])