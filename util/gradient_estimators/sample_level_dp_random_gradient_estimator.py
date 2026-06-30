from __future__ import annotations

from typing import Callable

import torch

from util.gradient_estimators.random_gradient_estimator import (
    RandomGradientEstimator,
    RandomGradEstimateMethod,
)


class SampleLevelDPRandomGradientEstimator(RandomGradientEstimator):
    """
    Prototype estimator for DPZero-style sample-level scalar DP.

    Difference from the main communication-level DP pipeline:
    - Main pipeline: compute one batch-level directional scalar, then clip/noise
      the scalar before communication.
    - This prototype: compute one directional scalar per sample, clip each
      per-sample scalar, average clipped scalars, then add Gaussian noise.

    The returned tensor still has shape [num_pert], so the existing server and
    aggregation code can remain unchanged.

    Important convention:
    sample_dp_sigma is the absolute Gaussian std added to the averaged clipped
    scalar. If you use an accountant that outputs a noise multiplier, convert it
    to this absolute std before passing it here.
    """

    def __init__(
        self,
        *args,
        sample_dp_clip_threshold: float = 1.0,
        sample_dp_sigma: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not self.paramwise_perturb:
            raise ValueError(
                "SampleLevelDPRandomGradientEstimator requires paramwise_perturb=True "
                "for full-model LLM ZO."
            )
        self.sample_dp_clip_threshold = float(sample_dp_clip_threshold)
        self.sample_dp_sigma = float(sample_dp_sigma)

    def compute_sample_level_dp_grad_scalars(
        self,
        batch_inputs,
        labels: torch.Tensor,
        per_sample_loss_fn: Callable[[object, torch.Tensor], torch.Tensor],
        model_inference: Callable[[object], object],
        seed: int,
    ) -> torch.Tensor:
        """
        Compute DPZero-style private directional scalars.

        Args:
            batch_inputs: model inputs for the current mini-batch.
            labels: labels for the current mini-batch.
            per_sample_loss_fn: callable returning shape [batch_size] losses.
            model_inference: zero-argument closure returning model output for
                the current batch_inputs.
            seed: seed used to regenerate perturbation directions.

        Returns:
            Tensor of shape [num_pert], where each entry is a noised averaged
            per-sample directional scalar for one perturbation direction.
        """
        if self.grad_estimate_method != RandomGradEstimateMethod.rge_central:
            raise NotImplementedError(
                "Sample-level DP prototype currently supports rge_central only."
            )

        dir_grads = []
        C = self.sample_dp_clip_threshold

        for perturb_idx in range(self.num_pert):
            # θ + μz
            rng = self.get_rng(seed, perturb_idx)
            self.perturb_model_paramwise(rng, alpha=self.mu)
            loss_plus_vec = per_sample_loss_fn(model_inference(), labels)

            # θ - μz
            rng = self.get_rng(seed, perturb_idx)
            self.perturb_model_paramwise(rng, alpha=-2 * self.mu)
            loss_minus_vec = per_sample_loss_fn(model_inference(), labels)

            # Restore θ
            rng = self.get_rng(seed, perturb_idx)
            self.perturb_model_paramwise(rng, alpha=self.mu)

            if loss_plus_vec.shape != loss_minus_vec.shape:
                raise ValueError(
                    "per_sample_loss_fn must return tensors with matching shapes "
                    f"for plus/minus losses, got {loss_plus_vec.shape} and {loss_minus_vec.shape}."
                )
            if loss_plus_vec.ndim != 1:
                raise ValueError(
                    "per_sample_loss_fn must return a 1-D tensor of shape [batch_size]."
                )

            sample_dir_scalars = (loss_plus_vec - loss_minus_vec) / (2 * self.mu)
            clipped = torch.clamp(sample_dir_scalars, min=-C, max=C)
            private_scalar = clipped.mean()

            if self.sample_dp_sigma > 0:
                private_scalar = private_scalar + torch.randn_like(private_scalar) * self.sample_dp_sigma

            dir_grads.append(private_scalar)

        return torch.stack(dir_grads).to(device=self.device)
