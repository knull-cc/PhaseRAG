import torch
import torch.nn.functional as F


def metric(pred: torch.Tensor, true: torch.Tensor) -> dict[str, torch.Tensor]:
    mse = F.mse_loss(pred, true)
    mae = torch.mean(torch.abs(pred - true))
    rmse = torch.sqrt(mse)
    mape = torch.mean(torch.abs(100 * (pred - true) / (true + 1e-8)))
    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
    }
