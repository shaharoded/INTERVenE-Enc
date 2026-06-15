"""
train.py
========

Fit one :class:`JumpReLUTranscoder` per encoder layer on cached MLP I/O pairs.

Loss
----
    L = MSE(W_dec . f(x) + b_dec , y_true)  +  lambda_l0 * L0(f)

where ``L0(f) = sum_i 1{f_i > 0}`` (Heaviside, STE'd through JumpReLU's
threshold gate).  No L1 -- we want sparsity *without* magnitude shrinkage on
the active features.
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from xai.transcoder.transcoder import JumpReLUTranscoder, _StepWithSTE


def _l0_surrogate(f: torch.Tensor) -> torch.Tensor:
    """
    Differentiable L0 = sum over features of Heaviside(f).

    JumpReLU produces exact zeros below threshold, so Heaviside is a clean
    indicator.  We re-wrap with the STE so theta still gets gradient from L0.
    """
    return _StepWithSTE.apply(f, 1e-3).sum(dim=-1).mean()


def train_one(transcoder: JumpReLUTranscoder, X: torch.Tensor, Y: torch.Tensor,
              n_epochs: int = 20, batch_size: int = 4096,
              lr: float = 3e-4, lambda_l0: float = 3e-4,
              device: str = "cpu", desc: str = "transcoder"):
    """
    Purpose: Train a single transcoder on (X, Y) activation pairs.
    Method:  AdamW on MSE + lambda_l0 * L0.

    Args:
        transcoder (JumpReLUTranscoder)
        X (Tensor [N, d_model]): MLP inputs.
        Y (Tensor [N, d_model]): MLP outputs (regression target).
        n_epochs   (int)
        batch_size (int)
        lr         (float)
        lambda_l0  (float): sparsity weight.
        device     (str)
        desc       (str): tqdm label.

    Returns:
        dict with epoch-level history: mse, l0, total.
    """
    transcoder = transcoder.to(device)
    ds = TensorDataset(X, Y)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    opt = torch.optim.AdamW(transcoder.parameters(), lr=lr)

    hist = {"mse": [], "l0": [], "total": []}
    for ep in range(n_epochs):
        tot_mse = tot_l0 = tot = 0.0
        n_steps = 0
        pbar = tqdm(dl, desc=f"{desc} ep{ep+1}/{n_epochs}", leave=False,
                    mininterval=1.0)
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)
            f, y_hat = transcoder(xb)
            mse = nn.functional.mse_loss(y_hat, yb)
            l0 = _l0_surrogate(f)
            loss = mse + lambda_l0 * l0
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot_mse += mse.item(); tot_l0 += l0.item(); tot += loss.item()
            n_steps += 1
        ep_mse = tot_mse / max(1, n_steps)
        ep_l0  = tot_l0  / max(1, n_steps)
        hist["mse"].append(ep_mse)
        hist["l0"].append(ep_l0)
        hist["total"].append(tot / max(1, n_steps))
        # Use tqdm.write so the line nests cleanly with any outer bar and does
        # not collide with the inner per-step bar (already closed by now).
        tqdm.write(f"  [{desc}] epoch {ep+1:3d}/{n_epochs}  "
                   f"mse={ep_mse:.4e}  L0={ep_l0:6.1f}")
    return hist


def train_transcoders(activations: dict, d_model: int, n_features: int,
                      n_epochs: int = 20, batch_size: int = 4096,
                      lr: float = 3e-4, lambda_l0: float = 3e-4,
                      bandwidth: float = 1e-3, init_theta: float = 0.1,
                      device: str = "cpu"):
    """
    Purpose: Fit one transcoder per captured layer.
    Method:  Iterate :func:`train_one` over ``activations`` dict from
             :func:`transcoder.hooks.collect_activations`.

    Args:
        activations (Dict[int, (X, Y)])
        d_model     (int)
        n_features  (int): dictionary size.
        ...

    Returns:
        Dict[int, JumpReLUTranscoder], Dict[int, history-dict].
    """
    models, histories = {}, {}
    n_layers = len(activations)
    for li, (layer_idx, (X, Y)) in enumerate(activations.items(), start=1):
        print(f"\n=== Transcoder {li}/{n_layers} — encoder layer {layer_idx} "
              f"(X={tuple(X.shape)}, n_features={n_features}) ===")
        tc = JumpReLUTranscoder(d_model=d_model, n_features=n_features,
                                bandwidth=bandwidth, init_theta=init_theta)
        h = train_one(tc, X, Y, n_epochs=n_epochs, batch_size=batch_size,
                      lr=lr, lambda_l0=lambda_l0, device=device,
                      desc=f"layer{layer_idx}")
        models[layer_idx] = tc.eval()
        histories[layer_idx] = h
        print(f"  -> done  final mse={h['mse'][-1]:.4e}  final L0={h['l0'][-1]:.1f}")
    return models, histories


def save_transcoders(models: dict, path):
    """Serialise a {layer_idx: JumpReLUTranscoder} bundle to one file."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "layers": sorted(models.keys()),
        "d_model": next(iter(models.values())).d_model,
        "n_features": next(iter(models.values())).n_features,
        "state": {i: m.state_dict() for i, m in models.items()},
    }
    torch.save(payload, path)


def load_transcoders(path, map_location: str = "cpu"):
    """Restore a bundle produced by :func:`save_transcoders`."""
    payload = torch.load(path, map_location=map_location, weights_only=True)
    out = {}
    for i in payload["layers"]:
        tc = JumpReLUTranscoder(d_model=payload["d_model"],
                                n_features=payload["n_features"])
        tc.load_state_dict(payload["state"][i])
        out[i] = tc.eval()
    return out
