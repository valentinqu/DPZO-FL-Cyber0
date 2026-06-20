from __future__ import annotations

import random
from collections import deque
from typing import Any, Callable, Iterable, Sequence

import torch

from .client import AbstractClient
from util.gradient_estimators.abstract_gradient_estimator import AbstractGradientEstimator
from .typing import CriterionType
from util.metrics import Metric

# Type alias
# OneRoundGradScalars contains a list (K local update) of (P perturbation) gradient scalars.
OneRoundGradScalars = list[torch.Tensor]
# MultiRoundGradScalars is a list of multiple OneRoundGradScalars for multiple rounds.
MultiRoundGradScalars = list[list[torch.Tensor]]
# MultiClientOneRoundGradScalars is a list of OneRoundGradScalars for multiple client
MultiClientOneRoundGradScalars = list[list[torch.Tensor]]

AggregationFunc = Callable[[MultiClientOneRoundGradScalars], OneRoundGradScalars]
AttackFunc = Callable[[MultiClientOneRoundGradScalars], MultiClientOneRoundGradScalars]


def fed_avg(
    local_grad_scalar_list: MultiClientOneRoundGradScalars,
) -> OneRoundGradScalars:
    num_sample_clients = len(local_grad_scalar_list)
    grad_scalar: list[torch.Tensor] = []
    for each_local_step_update in zip(*local_grad_scalar_list):
        grad_scalar.append(sum(each_local_step_update).div_(num_sample_clients))
    return grad_scalar


class SeedAndGradientRecords:
    def __init__(self) -> None:
        """
        For seed_records/grad_records, each entry stores info related to 1 iteration:
        - seed_records[i]: length = number of local updates K
        - seed_records[i][k]: seed_k
        - grad_records[i]: [vector for local_update_k for k in range(K)]
        - grad_records[i][k]: scalar for 1 perturb or vector for >=1 perturb

        In stateful DeComFL, clients can replay missed global updates from these records.
        Stateless full-model LLM clients simply ignore pull_model(); the server still uses
        the same record structure to keep the ZO seed/scalar update history consistent.
        """

        self.seed_records: deque[list[int]] = deque()
        self.grad_records: deque[list[torch.Tensor]] = deque()
        self.earliest_records = 0
        self.current_iteration = -1

    def add_records(self, seeds: list[int], grad: list[torch.Tensor]) -> int:
        self.current_iteration += 1
        self.seed_records.append(seeds)
        self.grad_records.append(grad)
        return self.current_iteration

    def remove_too_old(self, earliest_record_needs: int):
        if self.earliest_records >= earliest_record_needs:
            return
        while self.earliest_records < earliest_record_needs:
            self.seed_records.popleft()
            self.grad_records.popleft()
            self.earliest_records += 1

    def fetch_seed_records(self, earliest_record_needs: int) -> list[list[int]]:
        assert earliest_record_needs >= self.earliest_records
        return [
            self.seed_records[i - self.earliest_records]
            for i in range(earliest_record_needs, self.current_iteration + 1)
        ]

    def fetch_grad_records(self, earliest_record_needs: int) -> list[list[torch.Tensor]]:
        assert earliest_record_needs >= self.earliest_records
        return [
            self.grad_records[i - self.earliest_records]
            for i in range(earliest_record_needs, self.current_iteration + 1)
        ]


class CeZO_Server:
    def __init__(
        self,
        clients: Sequence[AbstractClient],
        device: torch.device,
        num_sample_clients: int = 10,
        local_update_steps: int = 10,
    ) -> None:
        self.clients = clients
        self.device = device
        self.num_sample_clients = num_sample_clients
        self.local_update_steps = local_update_steps

        self.seed_grad_records = SeedAndGradientRecords()
        self.client_last_updates = [0 for _ in range(len(self.clients))]

        self.server_model: torch.nn.Module | None = None
        self.server_model_inference: Callable[[torch.nn.Module, Any], torch.Tensor] | None = None
        self.server_criterion: CriterionType | None = None
        self.server_accuracy_func = None
        self.optim: torch.optim.Optimizer | None = None
        self.gradient_estimator: AbstractGradientEstimator | None = None

        self._aggregation_func: AggregationFunc = fed_avg
        self._attack_func: AttackFunc = lambda x: x

    def set_server_model_and_criterion(
        self,
        model: torch.nn.Module,
        model_inference: Callable[[torch.nn.Module, Any], torch.Tensor],
        criterion: CriterionType,
        accuracy_func,
        optimizer: torch.optim.Optimizer,
        gradient_estimator: AbstractGradientEstimator,
    ) -> None:
        self.server_model = model
        self.server_model_inference = model_inference
        self.server_criterion = criterion
        self.server_accuracy_func = accuracy_func
        self.optim = optimizer
        self.gradient_estimator = gradient_estimator

    def get_sampled_client_index(self) -> list[int]:
        n = min(self.num_sample_clients, len(self.clients))
        return random.sample(range(len(self.clients)), n)

    def set_perturbation(self, num_pert: int) -> None:
        for client in self.clients:
            client.gradient_estimator().num_pert = num_pert
        if self.gradient_estimator is not None:
            self.gradient_estimator.num_pert = num_pert

    def set_learning_rate(self, lr: float) -> None:
        # Stateful CNN clients have optimizers; stateless LLM clients may use optimizer=None.
        for client in self.clients:
            client_optim = getattr(client, "optimizer", None)
            if client_optim is not None:
                for p in client_optim.param_groups:
                    p["lr"] = lr

        if self.server_model and self.optim:
            for p in self.optim.param_groups:
                p["lr"] = lr

    def aggregation_func(
        self, local_grad_scalar_list: MultiClientOneRoundGradScalars
    ) -> OneRoundGradScalars:
        return self._aggregation_func(local_grad_scalar_list)

    def attack_func(
        self, local_grad_scalar_list: MultiClientOneRoundGradScalars
    ) -> MultiClientOneRoundGradScalars:
        return self._attack_func(local_grad_scalar_list)

    def register_aggregation_func(self, aggregation_func: AggregationFunc) -> None:
        self._aggregation_func = aggregation_func

    def register_attack_func(self, attack_func: AttackFunc) -> None:
        self._attack_func = attack_func

    def train_one_step(self, iteration: int) -> tuple[float, float]:
        assert self.gradient_estimator is not None and self.optim is not None

        sampled_client_index = self.get_sampled_client_index()
        seeds = [random.randint(0, 1000000) for _ in range(self.local_update_steps)]

        local_grad_scalar_list = []
        total_loss = 0.0
        total_acc = 0.0

        for index in sampled_client_index:
            client = self.clients[index]
            last_update_idx = self.client_last_updates[index]

            if iteration > 0:
                seeds_history = self.seed_grad_records.fetch_seed_records(last_update_idx)
                grads_history = self.seed_grad_records.fetch_grad_records(last_update_idx)
                client.pull_model(seeds_history, grads_history)

            self.client_last_updates[index] = iteration

            if hasattr(client, "current_round"):
                client.current_round = iteration

            result = client.local_update(seeds)

            local_grad_scalar_list.append(result.grad_tensors)
            total_loss += result.step_loss
            total_acc += result.step_accuracy

        local_grad_scalar_list = self.attack_func(local_grad_scalar_list)
        global_grad_scalar = self.aggregation_func(local_grad_scalar_list)

        self.seed_grad_records.add_records(seeds=seeds, grad=global_grad_scalar)
        self.seed_grad_records.remove_too_old(earliest_record_needs=min(self.client_last_updates))

        if self.server_model:
            # Keep eval mode for LLM ZO to avoid dropout randomness during subsequent scalar evaluation.
            self.server_model.eval()
            self.gradient_estimator.update_model_given_seed_and_grad(
                self.optim,
                seeds,
                global_grad_scalar,
            )
            self.gradient_estimator.update_gradient_estimator_given_seed_and_grad(
                seeds,
                global_grad_scalar,
            )

        return total_loss / len(sampled_client_index), total_acc / len(sampled_client_index)

    def eval_model(self, test_loader: Iterable[Any]) -> tuple[float, float]:
        assert (
            self.server_model
            and self.gradient_estimator
            and self.server_criterion
            and self.server_accuracy_func
            and self.server_model_inference
        )

        self.server_model.eval()
        eval_loss = Metric("Eval loss")
        eval_accuracy = Metric("Eval accuracy")
        with torch.no_grad():
            for _, (batch_inputs, batch_labels) in enumerate(test_loader):
                if hasattr(batch_inputs, "to"):
                    batch_inputs = batch_inputs.to(device=self.device)
                elif isinstance(batch_inputs, torch.Tensor):
                    batch_inputs = batch_inputs.to(self.device)

                if isinstance(batch_labels, torch.Tensor):
                    batch_labels = batch_labels.to(self.device)

                pred = self.server_model_inference(self.server_model, batch_inputs)
                eval_loss.update(self.server_criterion(pred, batch_labels))
                eval_accuracy.update(self.server_accuracy_func(pred, batch_labels))
        print(
            f"\nEvaluation(Iteration {self.seed_grad_records.current_iteration}): ",
            f"Evaluation Loss:{eval_loss.avg:.4f}, Accuracy:{eval_accuracy.avg * 100:.2f}%",
        )
        return eval_loss.avg, eval_accuracy.avg
