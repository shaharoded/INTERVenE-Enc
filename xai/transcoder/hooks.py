"""
hooks.py
========

Forward-hook helpers that capture every ``MLP`` block's (input, output) pairs
from a frozen :class:`InterveneEncoder` -- without editing the source module.

The encoder's AdaLN block applies the MLP as::

    x = x + gate_mlp * self.mlp(norm_x)

so the MLP input we want is ``norm_x`` (the modulated post-LN value) and the
MLP output we want is the raw ``self.mlp(norm_x)`` *before* the gate scaling
and residual addition.  Both are exactly what ``self.mlp.__call__`` sees and
returns, which means a single forward-pre + forward hook on each
``block.mlp`` captures the right tensors with zero source edits.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
from tqdm.auto import tqdm


@dataclass
class MLPIOBuffer:
    """
    Holds captured MLP inputs / outputs for one layer.

    Tensors are appended in micro-batches and concatenated lazily via
    :meth:`stack`.  Stored on CPU to avoid OOM during long capture runs.
    """
    inputs:  List[torch.Tensor] = field(default_factory=list)
    outputs: List[torch.Tensor] = field(default_factory=list)

    def stack(self):
        """Return (X_in, Y_out) as flat [N_tokens, d_model] CPU tensors."""
        X = torch.cat(self.inputs,  dim=0)
        Y = torch.cat(self.outputs, dim=0)
        return X, Y


@contextmanager
def capture_mlp_io(model, layer_indices=None, pad_mask_getter=None):
    """
    Purpose: Attach forward-pre + forward hooks on selected ``block.mlp``
             modules and yield one :class:`MLPIOBuffer` per layer.
    Method:  Pre-hook stores ``input[0]`` flattened over batch & time; the
             forward hook stores the matching output.  Padded positions are
             dropped if ``pad_mask_getter`` is supplied.

    Args:
        model            (InterveneEncoder): the frozen encoder.
        layer_indices    (Iterable[int]|None): which blocks to hook
                          (default: every block).
        pad_mask_getter  (callable|None): function ``() -> Tensor[B,T] (bool,
                          True at valid positions)``.  Called inside each
                          hook so the caller can stash the current batch's
                          pad mask between forward() calls.

    Yields:
        Dict[int, MLPIOBuffer] keyed by layer index.
    """
    if layer_indices is None:
        layer_indices = list(range(len(model.blocks)))

    buffers = {i: MLPIOBuffer() for i in layer_indices}
    handles = []

    def make_pre_hook(layer_idx):
        def _pre_hook(_module, inputs):
            # `inputs` is a tuple of positional args; MLP takes a single tensor.
            x_in = inputs[0].detach()
            _stash_for_layer[layer_idx] = x_in       # see post-hook
        return _pre_hook

    def make_post_hook(layer_idx):
        def _post_hook(_module, _inputs, output):
            y_out = output.detach()
            x_in  = _stash_for_layer.pop(layer_idx, None)
            if x_in is None:
                return
            # Flatten [B, T, D] -> [B*T, D] and drop PAD if a mask is available.
            x_flat = x_in.reshape(-1, x_in.shape[-1])
            y_flat = y_out.reshape(-1, y_out.shape[-1])
            if pad_mask_getter is not None:
                m = pad_mask_getter()
                if m is not None:
                    keep = m.reshape(-1).to(x_flat.device)
                    x_flat = x_flat[keep]
                    y_flat = y_flat[keep]
            buffers[layer_idx].inputs.append(x_flat.float().cpu())
            buffers[layer_idx].outputs.append(y_flat.float().cpu())
        return _post_hook

    _stash_for_layer: dict = {}

    try:
        for i in layer_indices:
            mlp = model.blocks[i].mlp
            handles.append(mlp.register_forward_pre_hook(make_pre_hook(i)))
            handles.append(mlp.register_forward_hook(make_post_hook(i)))
        yield buffers
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def collect_activations(model, dataloader, device, max_batches=None,
                        layer_indices=None):
    """
    Purpose: Run the frozen encoder over a dataloader and harvest
             (mlp_in, mlp_out) pairs from every requested layer.
    Method:  Wraps :func:`capture_mlp_io`; runs ``model.encode`` (no task
             heads needed) and filters out padded tokens via the returned
             pad mask.

    Args:
        model         (InterveneEncoder): frozen encoder (call .eval()).
        dataloader    (DataLoader)       : provides standard EMR batches.
        device        (torch.device)
        max_batches   (int|None)         : early-stop after N batches (smoke test).
        layer_indices (list[int]|None)

    Returns:
        Dict[int, (X_in, Y_out)] -- both CPU tensors of shape [N_tokens, d_model].
    """
    model.eval()
    pad_holder = {"mask": None}
    getter = lambda: pad_holder["mask"]

    pad_idx = model.embedder.padding_idx
    with capture_mlp_io(model, layer_indices=layer_indices,
                        pad_mask_getter=getter) as buffers:
        for bi, batch in enumerate(tqdm(dataloader, desc="capture", leave=False)):
            if max_batches is not None and bi >= max_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            # Set the pad mask BEFORE encode so the hooks see it (the encoder
            # forward fires the MLP hooks during, not after, the call).
            pad_holder["mask"] = (batch["position_ids"] != pad_idx)
            model.encode(
                parent_raw_ids=batch["parent_raw_ids"],
                concept_ids=batch["concept_ids"],
                value_ids=batch["value_ids"],
                position_ids=batch["position_ids"],
                abs_ts=batch["abs_ts"],
                context_vec=batch["context_vec"],
            )
        # Final flush.
        out = {i: buf.stack() for i, buf in buffers.items()}
    return out
