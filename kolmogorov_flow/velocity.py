import os
from typing import Callable

import numpy as np
import torch
from torchfsm.field import diffused_noise, kolm_force
from torchfsm.mesh import FourierMesh, MeshGrid
from torchfsm.operator import (
    Div,
    Laplacian,
    Operator,
    Vorticity2Velocity,
    VorticityConvection,
)
from torchfsm.traj_recorder import AutoRecorder, IntervalController
from tqdm.auto import tqdm


def Kolmogorov(force: Operator, Re=100) -> Operator:
    return -VorticityConvection() + 1 / Re * Laplacian() + force


def Divergence(
    device,
    dtype=None,
    resolution: int = 128,
) -> Callable:
    mesh = MeshGrid([(0, 2 * np.pi, resolution), (0, 2 * np.pi, resolution)], device=device)
    divergence = Div()
    divergence.register_mesh(mesh, n_channel=2)
    divergence.to(device=device, dtype=dtype)
    return divergence


def generate_dataset(
    save_dir: str,
    Res: list,
    resolution: int,
    num_steps=1500,
    record_interval=10,
    step_start_record=1001,
    device="cuda",
    batch_size: int = 1,
    dt_coef: float = 1,  # dt=1/Re*dt_coef
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mesh = MeshGrid([(0, 2 * np.pi, resolution), (0, 2 * np.pi, resolution)], device=device)
    x, y = mesh.bc_mesh_grid()
    force = kolm_force(y)
    vor_2_vel = Vorticity2Velocity()
    vor_2_vel.register_mesh(mesh, n_channel=1)
    for re in tqdm(Res):
        kolm = Kolmogorov(Re=re, force=force)
        kolm.register_mesh(mesh, n_channel=1)
        dt = 1 / re * dt_coef
        recorder = AutoRecorder(IntervalController(start=step_start_record, interval=record_interval))
        u = FourierMesh(mesh).fft(diffused_noise(mesh, n_batch=batch_size))
        for step in tqdm(range(num_steps), leave=False):
            u = kolm.integrate(u_0_fft=u, dt=dt, step=1, return_in_fourier=True)
            vel = vor_2_vel(u_fft=u, return_in_fourier=True)
            recorder.record(step + 1, vel)
        np.save(os.path.join(save_dir, f"Re_{re}_uv.npy"), recorder.trajectory.cpu().numpy())
