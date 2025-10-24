import argparse
import csv
import logging
import os
import shutil
from collections import OrderedDict
from copy import deepcopy
from time import time

import numpy as np
import torch
import torch.distributed as dist
from conflictfree.grad_operator import ConFIG_update
from conflictfree.utils import apply_gradient_vector, get_gradient_vector
from dataset_loader import DarcyDataset, k_mean, k_std, p_mean, p_std
from dit import DiT
from flow_matching import *
from grad_utils import GradientsHelper, generalized_b_xy_c_to_image
from matplotlib import pyplot as plt
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from unet3d import Unet3D

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

dtype = torch.float32


@torch.no_grad()
def update_ema(ema_model, model, decay=0.999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    dist.destroy_process_group()


def create_logger(logging_dir):
    if dist.get_rank() == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")],
        )
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, "Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, device={device}, seed={seed}, world_size={dist.get_world_size()}.")

    model_string_name = args.version
    results_dir = "logs"
    if rank == 0:
        os.makedirs(results_dir, exist_ok=True)
        experiment_dir = f"{results_dir}/{model_string_name}"
        shutil.rmtree(experiment_dir, ignore_errors=True)
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        f_csv = open(f"{experiment_dir}/validation.csv", "w", encoding="UTF8", newline="")
        writer_csv = csv.writer(f_csv)
        writer_csv.writerow(["epoch", "residual", "eval_loss"])
        f_csv.close()
    else:
        experiment_dir = f"{results_dir}/{model_string_name}"
        logger = create_logger(None)

    darcy_train = DarcyDataset(("train/p_data.csv", "train/K_data.csv"))
    darcy_valid = DarcyDataset(("valid/p_data.csv", "valid/K_data.csv"))

    train_sampler = DistributedSampler(darcy_train, num_replicas=dist.get_world_size(), rank=rank, shuffle=True, seed=args.global_seed)
    train_loader = DataLoader(
        darcy_train, batch_size=int(args.global_batch_size // dist.get_world_size()), shuffle=False, sampler=train_sampler, num_workers=7, persistent_workers=True, drop_last=True
    )

    valid_sampler = DistributedSampler(darcy_valid, num_replicas=dist.get_world_size(), rank=rank, shuffle=True, seed=args.global_seed)
    valid_loader = DataLoader(
        darcy_valid, batch_size=int(args.global_batch_size // dist.get_world_size()), shuffle=False, sampler=valid_sampler, num_workers=7, persistent_workers=True, drop_last=True
    )

    if args.use_unet:
        model = Unet3D(dim=32, channels=2, sigmoid_last_channel=False)
    else:
        model = DiT()

    model.to(device)
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[device], find_unused_parameters=True)

    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0)

    grad_helper = GradientsHelper(device=device)

    update_ema(ema, model.module, decay=0)
    model.train()
    ema.eval()

    train_steps = 0
    running_loss = 0
    running_loss_residual = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        n_steps = ((epoch - 1) // (args.epochs // 4) + 1) if args.use_unrolling else 1
        model.train()
        train_sampler.set_epoch(epoch)
        for x_1 in train_loader:
            x_1 = x_1.to(device)
            x_0 = torch.randn_like(x_1)
            t = sample_t(x_1)
            x_t = psi_t(x_0, x_1, t)
            v_t = u_t(x_0, x_1)
            model_kwargs = dict()
            loss, residual_loss = cfm_loss_residual(model.module, x_t, t, v_t, grad_helper, args.use_dignorm, n_steps, **model_kwargs)

            if args.use_residual:
                grads = []
                opt.zero_grad()
                loss.backward(retain_graph=True)
                grads.append(get_gradient_vector(model.module))
                opt.zero_grad()
                residual_loss.backward()
                grads.append(get_gradient_vector(model.module))
                if not grads[1].isnan().any():
                    g_config = ConFIG_update(grads)
                    apply_gradient_vector(model.module, g_config)
                else:
                    print("NaN detected")
                    apply_gradient_vector(model.module, grads[0])
            else:
                opt.zero_grad()
                loss.backward()

            opt.step()
            update_ema(ema, model.module)

            running_loss += loss.item()
            running_loss_residual += residual_loss.item()
            train_steps += 1

        if epoch % args.eval_every == 0:
            model.eval()
            for x_1 in valid_loader:
                x_1 = x_1.to(device)
                x_0 = torch.randn_like(x_1)
                t = sample_t(x_1)
                x_t = psi_t(x_0, x_1, t)
                v_t = u_t(x_0, x_1)
                model_kwargs = dict()
                eval_loss, _ = cfm_loss_residual(ema, x_t, t, v_t, grad_helper, args.use_dignorm, n_steps, **model_kwargs)
                dist.all_reduce(eval_loss, op=dist.ReduceOp.AVG)
                break

            residuals = []
            with torch.no_grad():
                for s in range(256 // dist.get_world_size()):
                    x_1 = darcy_train[s].to(device).unsqueeze(0)
                    x_0 = torch.randn_like(x_1)
                    x_1_pred = sample(ema, x_0, num_steps=args.fm_steps, use_stoc_samp=args.use_stoc_samp, **model_kwargs)
                    x_1_pred[:, 0] = x_1_pred[:, 0] * p_std + p_mean
                    x_1_pred[:, 1] = x_1_pred[:, 1] * k_std + k_mean

                    residual = grad_helper.compute_residual(x_1_pred)["residual"].abs()
                    residual = generalized_b_xy_c_to_image(residual)

                    if s == 0:
                        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
                        plt.subplots_adjust(wspace=0.5)

                        c0 = ax[0].imshow(x_1_pred[0, 1].cpu().numpy().T, cmap="magma")
                        c1 = ax[1].imshow(x_1_pred[0, 0].cpu().numpy().T, cmap="magma")
                        c2 = ax[2].imshow(residual[0, 0].cpu().numpy().T, cmap="magma", norm="log")
                        c_bar = [c0, c1, c2]

                        for k, axx in enumerate(ax):
                            axx.set_xlabel(r"$\xi 1$")
                            axx.set_ylabel(r"$\xi 2$")
                            axx.invert_yaxis()
                            axx.set_xticks([-0.5, 63.5])
                            axx.set_xticklabels([0, 1])
                            axx.set_yticks([-0.5, 63.5])
                            axx.set_yticklabels([0, 1])
                            axx.set_aspect("equal")

                            position = axx.get_position()
                            cax = fig.add_axes([position.x1 + 0.01, position.y0, 0.01, position.y1 - position.y0])
                            cb = plt.colorbar(c_bar[k], cax=cax, orientation="vertical")
                            cb.minorticks_on()

                        ax[0].set_title(r"Permeability $K$")
                        ax[1].set_title(r"Pressure $p$")
                        ax[2].set_title(r"Residual $R_{MAE}(K,p)$")

                        plt.savefig(f"{experiment_dir}/sample_{rank:02d}.svg", format="svg", bbox_inches="tight")
                        plt.close("all")

                        np.save(f"{experiment_dir}/sample_{rank:02d}.npy", x_1_pred.cpu().numpy())
                    residuals.append(residual.abs().mean().item())

            residuals = torch.tensor(residuals, device=device).mean()
            dist.all_reduce(residuals, op=dist.ReduceOp.AVG)
            dist.all_reduce(eval_loss, op=dist.ReduceOp.AVG)

            logger.info(f"Mean residual: {residuals.item():.4e}, Eval Loss: {eval_loss.item():.4e}")

            if rank == 0:
                f_csv = open(f"{experiment_dir}/validation.csv", "a", encoding="UTF8", newline="")
                writer_csv = csv.writer(f_csv)
                writer_csv.writerow([epoch, residuals.item(), eval_loss.item()])
                f_csv.close()

        if epoch % args.log_every == 0:
            torch.cuda.synchronize()
            end_time = time()
            sec_per_epoch = end_time - start_time
            avg_loss = torch.tensor(running_loss / train_steps, device=device)
            dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
            avg_loss_residual = torch.tensor(running_loss_residual / train_steps, device=device)
            dist.all_reduce(avg_loss_residual, op=dist.ReduceOp.AVG)
            logger.info(f"(epoch={epoch:06d}) Train Loss: {avg_loss:.4e}, Train loss residual: {avg_loss_residual:.4e}, Sec per epoch: {sec_per_epoch:.3e}")
            start_time = time()

        running_loss = 0
        running_loss_residual = 0
        train_steps = 0

        if epoch % args.ckpt_every == 0:
            if rank == 0:
                checkpoint = {"model": model.module.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "args": args}
                checkpoint_path = f"{checkpoint_dir}/{epoch:07d}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
            dist.barrier()

    model.eval()

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8000)
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--global-seed", type=int, default=15)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--ckpt-every", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--version", type=str, default="test")
    parser.add_argument("--fm_steps", type=int, default=20)
    parser.add_argument("--use-unet", type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--use-dignorm", type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--use-residual", type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--use-stoc-samp", type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--use-unrolling", type=bool, default=False, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    main(args)
