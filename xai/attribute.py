"""
attribute.py
============

Two-headed explainability:

1. **Swap-in hooks** -- replace each block's MLP output with the transcoder's
   sparse reconstruction at inference time.  Used to verify that the
   transcoders preserve downstream behaviour (logit fidelity, calibration) and
   to expose per-token, per-feature activations for plotting.

2. **Feature -> logit attribution** -- gradient x activation of an outcome
   logit w.r.t. each transcoder feature at every (layer, token) site.  This
   gives a linear-order estimate of which features the model relied on for a
   given prediction.

Neither path touches ``intervene_enc/`` source code; everything is wired up
through forward hooks and ``torch.autograd.grad``.
"""

from contextlib import contextmanager
from typing import Dict, List, Optional

import torch
import torch.nn as nn


class TranscoderHookManager:
    """
    Manage forward hooks that replace ``block.mlp`` output with a transcoder
    reconstruction, and optionally stash per-layer feature activations.

    Parameters
    ----------
    model        : InterveneEncoder (frozen).
    transcoders  : Dict[layer_idx -> JumpReLUTranscoder] -- e.g. from
                   :func:`transcoder.load_transcoders`.
    enabled      : whether the swap-in is active when hooks are attached.
                   When False the hooks become passive *sniffers* -- they
                   record features but leave the MLP output unchanged.
    """

    def __init__(self, model, transcoders: Dict[int, nn.Module],
                 enabled: bool = True):
        self.model = model
        self.transcoders = transcoders
        self.enabled = enabled
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        # Per-layer storage filled on every forward.  Tensors live on the
        # encoder's device so attribution can run autograd through them.
        self.features: Dict[int, torch.Tensor] = {}
        self.recon:    Dict[int, torch.Tensor] = {}
        self._track_grad = False

    # ------------------------------------------------------------ context ## #
    def attach(self, track_grad: bool = False):
        """
        Attach hooks.  When ``track_grad=True`` the stashed feature tensors
        ``requires_grad_()`` so :func:`feature_to_logit_attribution` can run
        autograd through them.
        """
        self._track_grad = track_grad
        for i, tc in self.transcoders.items():
            tc.to(next(self.model.parameters()).device).eval()
            h = self.model.blocks[i].mlp.register_forward_hook(self._make_hook(i))
            self._handles.append(h)
        return self

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self.features.clear(); self.recon.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.detach()

    # ---------------------------------------------------------- internals ## #
    def _make_hook(self, layer_idx: int):
        tc = self.transcoders[layer_idx]

        def _hook(_module, inputs, output):
            x_in = inputs[0]                                     # [B, T, D]
            f = tc.encode(x_in)                                  # [B, T, F]
            if self._track_grad:
                # Detach upstream so autograd treats f as a leaf, then re-decode
                # so the gradient w.r.t. f reflects only the path through this
                # MLP swap.
                f = f.detach().requires_grad_(True)
            y_hat = tc.decode(f)                                 # [B, T, D]
            self.features[layer_idx] = f
            self.recon[layer_idx]    = y_hat
            if self.enabled:
                return y_hat
            return output
        return _hook


@contextmanager
def feature_activations(model, transcoders, swap_in: bool = False):
    """
    Purpose: Convenience context manager -- yield a manager that records
             feature activations for every block on each forward pass.
    Method:  Wraps :class:`TranscoderHookManager` with sensible defaults.

    Args:
        model        (InterveneEncoder)
        transcoders  (Dict[int, JumpReLUTranscoder])
        swap_in      (bool): if True the MLP output is replaced by the
                             transcoder reconstruction; if False the model
                             runs untouched and the manager only sniffs.

    Yields:
        TranscoderHookManager
    """
    mgr = TranscoderHookManager(model, transcoders, enabled=swap_in).attach()
    try:
        yield mgr
    finally:
        mgr.detach()


def feature_to_logit_attribution(model, transcoders, batch, outcome_idx: int,
                                 patient_idx: int = 0):
    """
    Purpose: For one (patient, outcome), compute grad x activation of the
             risk logit w.r.t. each transcoder feature at every (layer, token).
    Method:  Attach hooks with ``track_grad=True`` so feature tensors are
             leaves; run :meth:`InterveneEncoder.predict`; pick the desired
             risk logit; backprop and multiply gradient by the feature value.

    Args:
        model        (InterveneEncoder): must have ``task_heads`` attached.
        transcoders  (Dict[int, JumpReLUTranscoder])
        batch        (dict): standard EMR batch already on the right device.
        outcome_idx  (int): index into ``task_heads.risk_idx``.
        patient_idx  (int): row of the batch to explain.

    Returns:
        Dict[int, Tensor[T, F]] per-layer attribution (signed), aligned with
        the original ``position_ids`` axis (PAD positions left in place; the
        caller can mask them with the standard pad-mask).
    """
    if model.task_heads is None:
        raise RuntimeError("attach_task_heads() before attribution.")

    # Gradient checkpointing in the encoder uses use_reentrant=True, which
    # disconnects the autograd graph when none of the checkpoint inputs require
    # grad -- exactly our situation (all model params are frozen for
    # attribution). The features tensor we stash inside the hook then never
    # propagates a gradient to risk_logits.  Disable checkpointing for the
    # duration of the attribution pass and restore it on the way out.
    prev_ckpt = getattr(model, "use_checkpoint", False)
    model.use_checkpoint = False

    mgr = TranscoderHookManager(model, transcoders, enabled=True).attach(track_grad=True)
    try:
        risk_logits, _, _, pad_mask = model.predict(
            parent_raw_ids=batch["parent_raw_ids"],
            concept_ids=batch["concept_ids"],
            value_ids=batch["value_ids"],
            position_ids=batch["position_ids"],
            abs_ts=batch["abs_ts"],
            context_vec=batch["context_vec"],
        )
        target = risk_logits[patient_idx, outcome_idx]
        # Backprop *just* into the feature leaves.
        leaves = list(mgr.features.values())
        grads = torch.autograd.grad(target, leaves, retain_graph=False,
                                    allow_unused=False)

        attribution = {}
        for (layer_idx, f), g in zip(mgr.features.items(), grads):
            # f, g have shape [B, T, F]; pick the patient row.
            attribution[layer_idx] = (f[patient_idx] * g[patient_idx]).detach()
        return attribution, pad_mask[patient_idx].detach()
    finally:
        mgr.detach()
        model.use_checkpoint = prev_ckpt
