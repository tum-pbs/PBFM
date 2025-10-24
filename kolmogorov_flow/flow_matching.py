import torch
from dataset_loader import scale_values
from velocity import Divergence

sig_min = 0.0


def psi_t(x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor, sig_min: float = sig_min) -> torch.Tensor:
    return (1 - (1 - sig_min) * t) * x_0 + t * x_1


def u_t(x_0: torch.Tensor, x_1: torch.Tensor, sig_min: float = sig_min) -> torch.Tensor:
    return x_1 - (1 - sig_min) * x_0


def sample_t(x_1: torch.tensor) -> torch.tensor:
    return torch.rand([x_1.shape[0]] + [1] * (x_1.dim() - 1), device=x_1.device)


def logit_normal_map(t: torch.Tensor, m: float = 0.0, s: float = 1.0) -> torch.Tensor:
    eps = 1e-6
    t = torch.clamp(t, eps, 1 - eps)
    return t * torch.exp(-0.5 * torch.square((torch.log(t / (1 - t)) - m) / s)) / (s * torch.sqrt(2 * torch.tensor(torch.pi)) * t * (1 - t) ** 2)


def cfm_loss_residual(model, x_t: torch.Tensor, t: torch.Tensor, v_t: torch.Tensor, use_dignorm: bool, n_time_steps: int, *args, **kwargs) -> torch.Tensor:
    u_t_pred = model(x_t, t.view(t.shape[0]), *args, **kwargs)

    t_1 = t.clone()
    dt = (1 - t) / n_time_steps
    x_1_pred = x_t + dt * u_t_pred
    for _ in range(1, n_time_steps):
        t_1 = t_1 + dt
        u_t_1 = model(x_1_pred, t_1.view(t_1.shape[0]), *args, **kwargs)
        x_1_pred = x_1_pred + dt * u_t_1
    residual = compute_residual(x_1_pred)
    error = torch.mean(residual, dim=(1, 2))
    loss_residual = torch.mean(t * error)
    w = logit_normal_map(t) if use_dignorm else torch.ones_like(t)
    loss = torch.mean(w * (u_t_pred - v_t) ** 2)
    return loss, loss_residual


def compute_residual(x: torch.Tensor) -> torch.Tensor:
    div = Divergence(device=x.device)

    x_scaled = torch.zeros_like(x)
    for i in range(len(scale_values)):
        x_scaled[:, i] = x[:, i] * scale_values[i]

    div_res = div(x_scaled[:, :2]).square()

    return div_res


def sample(ema, x_0: torch.tensor, num_steps: int, use_stoc_samp: bool, **kwargs) -> torch.tensor:
    time_steps = torch.linspace(0.0, 1.0, num_steps + 1, device=x_0.device)
    x_new = torch.clone(x_0)
    for t in torch.arange(0, num_steps, device=x_0.device):
        if time_steps[t] < 0.2 and use_stoc_samp:
            x_new += (1 - time_steps[t]) * ema(x_new, time_steps[t].unsqueeze(0), **kwargs)
            x_new = (1 - time_steps[t + 1]) * torch.randn_like(x_new) + time_steps[t + 1] * x_new
        else:
            x_new += (time_steps[t + 1] - time_steps[t]) * ema(x_new, time_steps[t].unsqueeze(0), **kwargs)

    return x_new
