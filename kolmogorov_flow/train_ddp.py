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
from dataset_loader import load_cpcf_test, load_cpcf_train, scale_values
from dit import DiT
from flow_matching import *
from matplotlib import pyplot as plt
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

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
    experiment_dir = f"{results_dir}/{model_string_name}"
    checkpoint_dir = f"{experiment_dir}/checkpoints"
    if rank == 0:
        os.makedirs(results_dir, exist_ok=True)
        shutil.rmtree(experiment_dir, ignore_errors=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        f_csv = open(f"{experiment_dir}/validation.csv", "w", encoding="UTF8", newline="")
        writer_csv = csv.writer(f_csv)
        writer_csv.writerow(["epoch", "residual", "u_m_loss", "u_s_loss", "v_m_loss", "v_s_loss"])
        f_csv.close()
    else:
        logger = create_logger(None)

    cp_cf_train = load_cpcf_train()
    cp_cf_valid = load_cpcf_test()

    train_sampler = DistributedSampler(cp_cf_train, num_replicas=dist.get_world_size(), rank=rank, shuffle=True, seed=args.global_seed)
    train_loader = DataLoader(
        cp_cf_train, batch_size=int(args.global_batch_size // dist.get_world_size()), shuffle=False, sampler=train_sampler, num_workers=7, persistent_workers=True, drop_last=True
    )

    valid_sampler = DistributedSampler(cp_cf_valid, num_replicas=dist.get_world_size(), rank=rank, shuffle=True, seed=args.global_seed)
    valid_loader = DataLoader(
        cp_cf_valid, batch_size=int(16 // dist.get_world_size()), shuffle=False, sampler=valid_sampler, num_workers=7, persistent_workers=True, drop_last=True
    )

    model = DiT()

    model.to(device)
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[device], find_unused_parameters=True)

    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

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
        for y, x_1 in train_loader:
            x_1 = x_1.to(device)
            y = y.to(device)
            x_0 = torch.randn_like(x_1)
            t = sample_t(x_1)
            x_t = psi_t(x_0, x_1, t)
            v_t = u_t(x_0, x_1)
            model_kwargs = dict(y=y)
            loss, residual_loss = cfm_loss_residual(model.module, x_t, t, v_t, args.use_dignorm, n_steps, **model_kwargs)

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
            for y, x_1 in valid_loader:
                x_1 = x_1.to(device)
                y = y.to(device)

                stat = 128

                y_repeat = y.repeat_interleave(stat, dim=0)
                model_kwargs = dict(y=y_repeat)

                x_0_repeat_shape = list(x_1.shape)
                x_0_repeat_shape.remove(2)
                x_0_repeat_shape[0] *= stat
                x_0 = torch.randn(torch.Size(x_0_repeat_shape), device=x_1.device)

                x_1_pred = sample(ema, x_0, num_steps=args.fm_steps, use_stoc_samp=args.use_stoc_samp, **model_kwargs)

                residuals = compute_residual(x_1_pred)
                residual = torch.mean(residuals)

                dist.all_reduce(residual, op=dist.ReduceOp.AVG)

                x_1_repeat_shape = list(x_1.shape)
                x_1_repeat_shape.remove(2)
                x_1_repeat_shape.insert(1, stat)

                x_1_pred_ms = torch.zeros_like(x_1)
                x_1_pred_ms[:, 0, :, :, :] = x_1_pred.view(x_1_repeat_shape).mean(dim=1)
                x_1_pred_ms[:, 1, :, :, :] = x_1_pred.view(x_1_repeat_shape).std(dim=1)

                u_m_loss = nn.functional.mse_loss(x_1[:, 0, 0, :, :], x_1_pred_ms[:, 0, 0, :, :])
                u_s_loss = nn.functional.mse_loss(x_1[:, 1, 0, :, :], x_1_pred_ms[:, 1, 0, :, :])
                v_m_loss = nn.functional.mse_loss(x_1[:, 0, 1, :, :], x_1_pred_ms[:, 0, 1, :, :])
                v_s_loss = nn.functional.mse_loss(x_1[:, 1, 1, :, :], x_1_pred_ms[:, 1, 1, :, :])

                dist.all_reduce(u_m_loss, op=dist.ReduceOp.AVG)
                dist.all_reduce(u_s_loss, op=dist.ReduceOp.AVG)
                dist.all_reduce(v_m_loss, op=dist.ReduceOp.AVG)
                dist.all_reduce(v_s_loss, op=dist.ReduceOp.AVG)

                for i in range(x_1.shape[2]):
                    x_1[:, 0, i, :, :] = x_1[:, 0, i, :, :] * scale_values[i]
                    x_1[:, 1, i, :, :] = x_1[:, 1, i, :, :] * scale_values[i]
                    x_1_pred_ms[:, 0, i, :, :] = x_1_pred_ms[:, 0, i, :, :] * scale_values[i]
                    x_1_pred_ms[:, 1, i, :, :] = x_1_pred_ms[:, 1, i, :, :] * scale_values[i]
                    x_1_pred[:, i, :, :] = x_1_pred[:, i, :, :] * scale_values[i]

                for s in range(x_1.shape[0]):
                    fig, ax = plt.subplots(nrows=4, ncols=x_1_pred_ms.shape[2], figsize=(8, 16))

                    for j, cmap in enumerate(["RdYlBu", "RdYlGn"]):

                        c1 = ax[0, j].imshow(x_1[s, 0, j, :, :].cpu().numpy().T, cmap=cmap)
                        c2 = ax[1, j].imshow(x_1[s, 1, j, :, :].cpu().numpy().T, cmap="Greys")
                        c3 = ax[2, j].imshow(x_1_pred_ms[s, 0, j, :, :].cpu().numpy().T, cmap=cmap)
                        c4 = ax[3, j].imshow(x_1_pred_ms[s, 1, j, :, :].cpu().numpy().T, cmap="Greys")

                        c_bar = [c1, c2, c3, c4]

                        for k in range(ax.shape[0]):
                            position = ax[k, j].get_position()
                            cax = fig.add_axes([position.x0, position.y0 - 0.02, position.x1 - position.x0, 0.01])
                            cb = plt.colorbar(c_bar[k], cax=cax, orientation="horizontal")
                            cb.minorticks_on()

                    ax[0, 0].set_ylabel("Ref mean")
                    ax[1, 0].set_ylabel("Ref std")
                    ax[2, 0].set_ylabel("Pred mean")
                    ax[3, 0].set_ylabel("Pred std")

                    ax[0, 0].set_title(r"$u$")
                    ax[0, 1].set_title(r"$v$")

                    for aax in ax.flatten():
                        aax.set_xticks([])
                        aax.set_yticks([])

                    plt.savefig(f"{experiment_dir}/sample_{rank*x_1.shape[0]+s:02d}.svg", format="svg", bbox_inches="tight")
                    plt.close("all")

                    np.save(f"{experiment_dir}/sample_{rank*x_1.shape[0]+s:02d}.npy", x_1_pred.view(x_1_repeat_shape).cpu().numpy())
                    np.save(f"{experiment_dir}/sample_{rank*x_1.shape[0]+s:02d}_ms.npy", x_1_pred_ms.cpu().numpy())

                if rank == 0:
                    f_csv = open(f"{experiment_dir}/validation.csv", "a", encoding="UTF8", newline="")
                    writer_csv = csv.writer(f_csv)
                    writer_csv.writerow([epoch, residual.item(), u_m_loss.item(), u_s_loss.item(), v_m_loss.item(), v_s_loss.item()])
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
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--global-seed", type=int, default=15)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--ckpt-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--version", type=str, default="test")
    parser.add_argument("--fm_steps", type=int, default=20)
    parser.add_argument("--use-dignorm", type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--use-residual", type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--use-stoc-samp", type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--use-unrolling", type=bool, default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    main(args)
