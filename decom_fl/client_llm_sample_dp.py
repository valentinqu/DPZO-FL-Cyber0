from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence, Callable, Any

import torch
from torch.utils.data import DataLoader

from util.metrics import Metric
from util.gradient_estimators.sample_level_dp_random_gradient_estimator import (
    SampleLevelDPRandomGradientEstimator,
)


@dataclass
class LocalUpdateResult:
    grad_tensors: list[torch.Tensor]
    step_accuracy: float
    step_loss: float

    def to(self, device: torch.device) -> "LocalUpdateResult":
        self.grad_tensors = [grad_tensor.to(device) for grad_tensor in self.grad_tensors]
        return self


class SampleLevelDPClient:
    """
    Stateless LLM client for a DPZero-style sample-level DP prototype.

    The server interface is unchanged: this client returns one scalar tensor per
    local ZO step, so the existing FOE and CyBeR-0 / Trimmed Mean server logic
    can still be used.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        model_inference: Callable[[torch.nn.Module, Any], Any],
        dataloader: DataLoader,
        grad_estimator: SampleLevelDPRandomGradientEstimator,
        criterion: Callable[[Any, torch.Tensor], torch.Tensor],
        per_sample_criterion: Callable[[Any, torch.Tensor], torch.Tensor],
        accuracy_func: Callable[[Any, torch.Tensor], torch.Tensor],
        device: torch.device,
        client_id: int | None = None,
        scalar_stats_logger=None,
    ):
        self.model = model
        self.model_inference = model_inference
        self.dataloader = dataloader
        self.grad_estimator = grad_estimator
        self.criterion = criterion
        self.per_sample_criterion = per_sample_criterion
        self.accuracy_func = accuracy_func
        self.device = device
        self.client_id = client_id
        self.scalar_stats_logger = scalar_stats_logger
        self.current_round: int | None = None
        self.optimizer = None
        self.data_iterator = self._get_train_batch_iterator()

    def gradient_estimator(self):
        return self.grad_estimator

    def _get_train_batch_iterator(self) -> Iterator:
        while True:
            for batch in self.dataloader:
                yield batch

    def _move_batch_to_device(self, batch_inputs, labels):
        if hasattr(batch_inputs, "to"):
            batch_inputs = batch_inputs.to(device=self.device)
        elif isinstance(batch_inputs, torch.Tensor):
            batch_inputs = batch_inputs.to(self.device)
        if isinstance(labels, torch.Tensor):
            labels = labels.to(self.device)
        return batch_inputs, labels

    def pull_model(
        self,
        seeds_list: Sequence[Sequence[int]],
        gradient_scalar: Sequence[Sequence[torch.Tensor]],
    ) -> None:
        # Stateless client: always evaluates the current shared global model.
        return

    def local_update(self, seeds: Sequence[int]) -> LocalUpdateResult:
        self.model.eval()
        train_loss = Metric("Sample-level DP client train loss")
        train_accuracy = Metric("Sample-level DP client train accuracy")
        iteration_grad_scalars: list[torch.Tensor] = []

        with torch.no_grad():
            for local_step_idx, seed in enumerate(seeds):
                batch_inputs, labels = next(self.data_iterator)
                batch_inputs, labels = self._move_batch_to_device(batch_inputs, labels)

                def inference_closure():
                    return self.model_inference(self.model, batch_inputs)

                grad_scalars = self.grad_estimator.compute_sample_level_dp_grad_scalars(
                    batch_inputs=batch_inputs,
                    labels=labels,
                    per_sample_loss_fn=self.per_sample_criterion,
                    model_inference=inference_closure,
                    seed=seed,
                    scalar_stats_logger=self.scalar_stats_logger,
                    round_idx=self.current_round,
                    client_id=self.client_id,
                    local_step_idx=local_step_idx,
                )
                iteration_grad_scalars.append(grad_scalars)

                pred = inference_closure()
                train_loss.update(self.criterion(pred, labels))
                train_accuracy.update(self.accuracy_func(pred, labels))

        return LocalUpdateResult(
            grad_tensors=iteration_grad_scalars,
            step_accuracy=train_accuracy.avg,
            step_loss=train_loss.avg,
        )
