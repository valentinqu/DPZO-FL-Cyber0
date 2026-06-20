import csv
import os
from pathlib import Path
from typing import Iterable

import torch


class RawScalarStatsLogger:
    """
    记录 Zero-Order 训练中客户端产生的 raw grad_scalars。

    注意：
    这里记录的是 DP clipping / Gaussian noise 之前的 scalars。
    用途是选择合适的 dp_clip_threshold C。
    """

    def __init__(
        self,
        output_path: str = "output/languages/scalar_stats/raw_scalar_values.csv",
        candidate_cs: Iterable[float] = (10, 20, 30, 50, 75, 100, 150, 200, 300, 500),
        max_vectors: int | None = None,
    ):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.candidate_cs = list(candidate_cs)
        self.max_vectors = max_vectors
        self.num_vectors = 0

        self.abs_values: list[float] = []

        file_exists = self.output_path.exists()
        self.file = open(self.output_path, "a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(
            self.file,
            fieldnames=[
                "round",
                "client_id",
                "local_step",
                "perturb_idx",
                "value",
                "abs_value",
            ],
        )

        if not file_exists or os.path.getsize(self.output_path) == 0:
            self.writer.writeheader()
            self.file.flush()

    def log(
        self,
        scalars: torch.Tensor,
        *,
        round_idx: int | None = None,
        client_id: int | None = None,
        local_step: int | None = None,
    ) -> None:
        if self.max_vectors is not None and self.num_vectors >= self.max_vectors:
            return

        x = scalars.detach().flatten().float().cpu()

        for perturb_idx, value in enumerate(x.tolist()):
            abs_value = abs(float(value))
            self.abs_values.append(abs_value)

            self.writer.writerow(
                {
                    "round": round_idx,
                    "client_id": client_id,
                    "local_step": local_step,
                    "perturb_idx": perturb_idx,
                    "value": float(value),
                    "abs_value": abs_value,
                }
            )

        self.num_vectors += 1

        if self.num_vectors % 20 == 0:
            self.file.flush()

    def print_summary(self) -> None:
        if not self.abs_values:
            print("[RawScalarStatsLogger] No scalar values collected.")
            return

        x = torch.tensor(self.abs_values, dtype=torch.float32)

        q_list = torch.tensor(
            [0.50, 0.75, 0.90, 0.95, 0.975, 0.99, 0.995],
            dtype=torch.float32,
        )
        q_values = torch.quantile(x, q_list)

        print("\n================ Raw ZO Scalar Statistics ================")
        print(f"num_values = {x.numel()}")
        print(f"mean       = {x.mean().item():.6f}")
        print(f"std        = {x.std(unbiased=False).item():.6f}")
        print(f"max        = {x.max().item():.6f}")

        for q, v in zip(q_list.tolist(), q_values.tolist()):
            print(f"p{q * 100:>5.1f}     = {v:.6f}")

        print("\nCandidate clipping thresholds:")
        for c in self.candidate_cs:
            clip_ratio = (x > c).float().mean().item()
            print(f"  C={c:<8.2f} clip_ratio={clip_ratio * 100:>6.2f}%")

        recommended_p95 = q_values[3].item()
        recommended_p975 = q_values[4].item()

        print("\nSuggested starting points:")
        print(f"  conservative utility C ≈ p97.5 = {recommended_p975:.6f}")
        print(f"  stronger clipping     C ≈ p95   = {recommended_p95:.6f}")
        print("===========================================================\n")

    def close(self) -> None:
        self.file.flush()
        self.file.close()