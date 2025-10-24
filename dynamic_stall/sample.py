import argparse
from time import time

import numpy as np
import torch
from dataset_loader import load_cpcf_test, mean_values, std_values
from dit import DiT
from flow_matching import *
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

dtype = torch.float32


def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    epoch = args.epoch
    results_dir = "logs"
    experiment_dir = f"{results_dir}/{args.version}"
    checkpoint_dir = f"{experiment_dir}/checkpoints"
    checkpoint_path = f"{checkpoint_dir}/{epoch:07d}.pt"

    cp_cf_valid = load_cpcf_test(full_conditioning=False)
    n_fields = cp_cf_valid[0][1].shape[1]
    valid_loader = DataLoader(cp_cf_valid, batch_size=1, shuffle=False)

    model = DiT()
    model.to(device)
    model.load_state_dict(torch.load(checkpoint_path)["ema"])
    model.eval()

    stat = 128
    with torch.no_grad():
        for step in [20]:
            all_pred_ms = np.zeros((len(cp_cf_valid), 2, n_fields, 128, 128))
            all_pred = np.zeros((len(cp_cf_valid) * stat, n_fields, 128, 128))
            losses = np.zeros((2, n_fields))

            time_tot = 0.0
            for idx, [y, x_1, _] in enumerate(valid_loader):
                x_1 = x_1.to(device)
                y = y.to(device)

                y_repeat = y.repeat_interleave(stat, dim=0)
                model_kwargs = dict(y=y_repeat)

                x_0_repeat_shape = list(x_1.shape)
                x_0_repeat_shape.remove(2)
                x_0_repeat_shape[0] *= stat
                x_0 = torch.randn(torch.Size(x_0_repeat_shape), device=x_1.device)

                time_start = time()
                x_1_pred = sample(model, x_0, num_steps=step, use_stoc_samp=args.use_stoc_samp, **model_kwargs)
                time_tot += time() - time_start

                x_1_repeat_shape = list(x_1.shape)
                x_1_repeat_shape.remove(2)
                x_1_repeat_shape.insert(1, stat)

                x_1_pred_ms = torch.zeros_like(x_1)
                x_1_pred_ms[:, 0, :, :, :] = x_1_pred.view(x_1_repeat_shape).mean(dim=1)
                x_1_pred_ms[:, 1, :, :, :] = x_1_pred.view(x_1_repeat_shape).std(dim=1)

                for i in range(losses.shape[0]):
                    for j in range(losses.shape[1]):
                        losses[i, j] += nn.functional.mse_loss(x_1[:, i, j, :, :], x_1_pred_ms[:, i, j, :, :])

                for i in range(x_1.shape[2]):
                    x_1[:, 0, i, :, :] = x_1[:, 0, i, :, :] * std_values[i] + mean_values[i]
                    x_1[:, 1, i, :, :] = x_1[:, 1, i, :, :] * std_values[i]
                    x_1_pred_ms[:, 0, i, :, :] = x_1_pred_ms[:, 0, i, :, :] * std_values[i] + mean_values[i]
                    x_1_pred_ms[:, 1, i, :, :] = x_1_pred_ms[:, 1, i, :, :] * std_values[i]
                    x_1_pred[:, i, :, :] = x_1_pred[:, i, :, :] * std_values[i] + mean_values[i]

                all_pred_ms[idx] = x_1_pred_ms.cpu().numpy().squeeze()
                all_pred[idx * stat : (idx + 1) * stat] = x_1_pred.cpu().numpy()

            losses /= len(cp_cf_valid)
            time_tot /= len(cp_cf_valid)

            print(f"Step: {step:03d}, Time: {time_tot:.3e}, Losses: " + " ".join("%.3e" % i for i in losses.flatten()))
            np.save(f"{experiment_dir}/sample_fm_{step:03d}.npy", all_pred)
            np.save(f"{experiment_dir}/sample_fm_{step:03d}_ms.npy", all_pred_ms)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=32000)
    parser.add_argument("--version", type=str, default="PBFM")
    parser.add_argument("--use-stoc-samp", type=bool, default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    main(args)
