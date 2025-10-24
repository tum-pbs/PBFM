import argparse

import numpy as np
import torch
from dataset_loader import DarcyDataset, k_mean, k_std, p_mean, p_std
from dit import DiT
from flow_matching import *
from grad_utils import GradientsHelper
from unet3d import Unet3D

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

dtype = torch.float32


def main(args):
    darcy_valid = DarcyDataset(("valid/p_data.csv", "valid/K_data.csv"))
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    grad_helper = GradientsHelper(device=device)

    epoch = args.epoch
    results_dir = "logs"
    experiment_dir = f"{results_dir}/{args.version}"
    checkpoint_dir = f"{experiment_dir}/checkpoints"
    checkpoint_path = f"{checkpoint_dir}/{epoch:07d}.pt"

    if args.use_unet:
        model = Unet3D(dim=32, channels=2, sigmoid_last_channel=False)
    else:
        model = DiT()
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path)["ema"])
    model.eval()

    batch_size = 32
    with torch.no_grad():
        for step in [20]:
            residuals = np.array([], dtype=np.float32)
            x_1_all = np.zeros((0, 2, 64, 64), dtype=np.float32)
            for s in range(1024 // batch_size):
                x_1 = darcy_valid[s].to(device).unsqueeze(0).repeat(batch_size, 1, 1, 1)
                x_0 = torch.randn_like(x_1)
                x_1_pred = sample(model, x_0, num_steps=step, use_stoc_samp=args.use_stoc_samp, **dict())
                x_1_pred[:, 0] = x_1_pred[:, 0] * p_std + p_mean
                x_1_pred[:, 1] = x_1_pred[:, 1] * k_std + k_mean

                x_1_all = np.concatenate((x_1_all, x_1_pred.cpu().numpy()), axis=0)

                residual = grad_helper.compute_residual(x_1_pred)["residual"].abs()

                residual = residual.abs().mean((1, 2)).cpu().numpy()
                residuals = np.concatenate((residuals, residual), axis=0)
            np.save(f"{experiment_dir}/sample_residual_{step:03d}.npy", residuals)
            print(step, residuals.mean())

            np.save(f"{experiment_dir}/samples_{step:03d}.npy", x_1_all)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=8000)
    parser.add_argument("--version", type=str, default="PBFM")
    parser.add_argument("--use-unet", type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--use-stoc-samp", type=bool, default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    main(args)
