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

        What should happen on clients pull server using grad_records[i][k]:
        - client use seed_records[i][k] to generate perturbation(s)
        - client_grad[i][k]: vector = mean(perturbations[j] * grad_records[i][k][j] for j)
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
            return  # No need to do anything
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
        self._attack_func: AttackFunc = lambda x: x  # No attach

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
        return random.sample(range(len(self.clients)), self.num_sample_clients)

    def set_perturbation(self, num_pert: int) -> None:
        for client in self.clients:
            client.gradient_estimator().num_pert = num_pert

    def set_learning_rate(self, lr: float) -> None:
        # Client
        for client in self.clients:
            for p in client.optimizer.param_groups:
                p["lr"] = lr
        # Server
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
        # TODO add function signature check
        self._aggregation_func = aggregation_func

    def register_attack_func(self, attack_func: AttackFunc) -> None:
        # TODO add function signature check
        self._attack_func = attack_func

    def train_one_step(self, iteration: int) -> tuple[float, float]:
        # 1. 采样客户端
        sampled_client_index = self.get_sampled_client_index()
        
        # 2. 生成本轮所有的随机种子 (Server 下发种子)
        # 这里的 seeds 是给 Client 本地训练每一步用的
        seeds = [random.randint(0, 1000000) for _ in range(self.local_update_steps)]

        # 3. 核心循环：拉取模型 + 本地训练
        # (替代了 execute_sampled_clients)
        local_grad_scalar_list = []
        total_loss = 0.0
        total_acc = 0.0
        
        for index in sampled_client_index:
            client = self.clients[index]
            
            # --- A. Client Pull (重放) ---
            # Client 需要从 Server 获取“由于自己上次更新以来”错过的所有历史记录
            last_update_idx = self.client_last_updates[index]
            
            # 从账本中获取历史种子和标量
            # fetch_seed_records 需要你在 SeedAndGradientRecords 类里确认是否是从 idx 开始取
            # 假设 fetch 逻辑是取 [last_update_idx, current_iteration] 之间的记录
            if iteration > 0: # 第一轮不需要 pull
                # 注意：SeedAndGradientRecords 的实现细节可能需要微调
                # 这里假设 fetch 接口接收的是“我需要的起始索引”
                seeds_history = self.seed_grad_records.fetch_seed_records(last_update_idx)
                grads_history = self.seed_grad_records.fetch_grad_records(last_update_idx)
                
                # 执行 Pull (Replay)
                client.pull_model(seeds_history, grads_history)
            
            # 更新该 Client 的最后同步时间
            self.client_last_updates[index] = iteration

            # --- B. Client Local Update (计算标量) ---
            # 这一步只算标量，不更新本地模型
            result = client.local_update(seeds)
            
            # 收集结果
            local_grad_scalar_list.append(result.grad_tensors)
            total_loss += result.step_loss
            total_acc += result.step_accuracy

        # 4. 聚合 (Aggregation)
        # 对所有 Client 返回的标量列表取平均
        local_grad_scalar_list = self.attack_func(local_grad_scalar_list)
        global_grad_scalar = self.aggregation_func(local_grad_scalar_list)

        # 5. 记录账本 (Record)
        self.seed_grad_records.add_records(seeds=seeds, grad=global_grad_scalar)
        self.seed_grad_records.remove_too_old(earliest_record_needs=min(self.client_last_updates))

        # 6. Server 更新全局模型 (Replay Global Model)
        if self.server_model:
            self.server_model.train()
            # 利用种子和聚合后的标量，重构梯度并更新 Server 模型
            self.gradient_estimator.update_model_given_seed_and_grad(
                self.optim,
                seeds,
                global_grad_scalar,
            )
            # 同时也更新 Estimator 的内部状态 (如果是 AdamForward 需要这个)
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
                if (
                    self.device != torch.device("cpu")
                    or self.gradient_estimator.torch_dtype != torch.float32
                ):
                    batch_inputs = batch_inputs.to(self.device, self.gradient_estimator.torch_dtype)
                    # In generation mode, labels are not tensor.
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
