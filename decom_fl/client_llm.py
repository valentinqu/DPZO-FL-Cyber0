from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Iterator, Sequence, Callable, Any

import torch
from torch.utils.data import DataLoader

from util.gradient_estimators.abstract_gradient_estimator import AbstractGradientEstimator
from util.gradient_estimators.random_gradient_estimator import RandomGradientEstimator
from .typing import CriterionType
from util.metrics import Metric


@dataclass
class LocalUpdateResult:
    grad_tensors: list[torch.Tensor]
    step_accuracy: float
    step_loss: float

    def to(self, device: torch.device) -> LocalUpdateResult:
        self.grad_tensors = [grad_tensor.to(device) for grad_tensor in self.grad_tensors]
        return self


class AbstractClient:
    device: torch.device

    @abc.abstractmethod
    def local_update(self, seeds: Sequence[int]) -> LocalUpdateResult:
        """Return one directional-scalar tensor for each local ZO step."""
        return NotImplemented

    @abc.abstractmethod
    def pull_model(
        self,
        seeds_list: Sequence[Sequence[int]],
        gradient_scalar: Sequence[Sequence[torch.Tensor]],
    ) -> None:
        """Synchronize client state with server history if the client is stateful."""
        ...

    @abc.abstractmethod
    def gradient_estimator(self) -> AbstractGradientEstimator:
        return NotImplemented


class ResetClient(AbstractClient):
    """
    Stateless LLM client for full-model forward-only ZO.

    This class intentionally does NOT keep a local model copy, LoRA adapter state, or
    optimizer state. It only evaluates directional scalar values on the current global
    model and returns those scalars to the server.

    Design goal:
    - closer to CyBeR-0 / MeZO-style federated ZO;
    - no LoRA matrices;
    - no backpropagation;
    - no client-side full-model transmission.

    The shared model is perturbed in-place during scalar estimation and restored by the
    ZO estimator before returning.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        model_inference: Callable[[torch.nn.Module, Any], torch.Tensor],
        dataloader: DataLoader,
        grad_estimator: AbstractGradientEstimator,
        optimizer: torch.optim.Optimizer | None,
        criterion: CriterionType,
        accuracy_func,
        device: torch.device,
        client_id: int | None = None,
        scalar_stats_logger=None,
    ):
        self.model = model
        self.model_inference = model_inference
        self.dataloader = dataloader
        self.device = device
        self.grad_estimator = grad_estimator
        self.optimizer = optimizer  # Kept only for compatibility; not used by this client.
        self.criterion = criterion
        self.accuracy_func = accuracy_func

        self.client_id = client_id
        self.current_round: int | None = None
        self.scalar_stats_logger = scalar_stats_logger

        self.data_iterator = self._get_train_batch_iterator()

    def gradient_estimator(self) -> AbstractGradientEstimator:
        return self.grad_estimator

    def _get_train_batch_iterator(self) -> Iterator:
        while True:
            for v in self.dataloader:
                yield v

    def _loss_fn(self, batch_inputs, batch_labels):
        return self.criterion(self.model_inference(self.model, batch_inputs), batch_labels)

    def _move_batch_to_device(self, batch_inputs, labels):
        if hasattr(batch_inputs, "to"):
            batch_inputs = batch_inputs.to(device=self.device)
        elif isinstance(batch_inputs, torch.Tensor):
            batch_inputs = batch_inputs.to(self.device)

        if isinstance(labels, torch.Tensor):
            labels = labels.to(self.device)

        return batch_inputs, labels

    def local_update(self, seeds: Sequence[int]) -> LocalUpdateResult:
        iteration_local_update_grad_vectors: list[torch.Tensor] = []
        train_loss = Metric("LLM client train loss")
        train_accuracy = Metric("LLM client train accuracy")

        # ZO finite differences must be deterministic; disable dropout.
        self.model.eval()

        if not isinstance(self.grad_estimator, RandomGradientEstimator):
            raise ValueError(
                "Stateless full-model LLM ZO currently supports RandomGradientEstimator only."
            )
        if not self.grad_estimator.paramwise_perturb:
            raise ValueError(
                "Full-model LLM ZO must use paramwise_perturb=True to avoid allocating "
                "a full perturbation vector or full gradient tensor."
            )

        with torch.no_grad():
            for local_step_idx, seed in enumerate(seeds):
                batch_inputs, labels = next(self.data_iterator)
                batch_inputs, labels = self._move_batch_to_device(batch_inputs, labels)

                # Compute only directional scalar values. Do NOT call optimizer.step(),
                # and do NOT materialize full gradients on the client.
                grad_scalars = self.grad_estimator._zo_grad_estimate_paramwise(
                    batch_inputs,
                    labels,
                    self._loss_fn,
                    seed,
                )

                # Log raw scalars before DP clipping/noising, if requested.
                if self.scalar_stats_logger is not None:
                    self.scalar_stats_logger.log(
                        grad_scalars,
                        round_idx=self.current_round,
                        client_id=self.client_id,
                        local_step=local_step_idx,
                    )

                # Communication-level DP-style scalar perturbation.
                # This protects uploaded directional scalar messages.
                if hasattr(self, "dpzero_clip_threshold") and hasattr(self, "dpzero_sigma"):
                    C = self.dpzero_clip_threshold
                    clip_factor = torch.clamp_max(
                        C / (torch.abs(grad_scalars) + 1e-8),
                        1.0,
                    )
                    grad_scalars = grad_scalars * clip_factor

                    if self.dpzero_sigma > 0:
                        noise = torch.randn_like(grad_scalars) * self.dpzero_sigma
                        grad_scalars = grad_scalars + noise

                iteration_local_update_grad_vectors.append(grad_scalars)

                # Training metrics are computed on the unperturbed model after the
                # finite-difference estimator has restored parameters.
                pred = self.model_inference(self.model, batch_inputs)
                train_loss.update(self.criterion(pred, labels))
                train_accuracy.update(self.accuracy_func(pred, labels))

        return LocalUpdateResult(
            grad_tensors=iteration_local_update_grad_vectors,
            step_accuracy=train_accuracy.avg,
            step_loss=train_loss.avg,
        )

    def pull_model(
        self,
        seeds_list: Sequence[Sequence[int]],
        gradient_scalar: Sequence[Sequence[torch.Tensor]],
    ) -> None:
        # Stateless full-model LLM clients always evaluate the current shared global
        # model. No client-local model or optimizer state needs to be replayed.
        return
