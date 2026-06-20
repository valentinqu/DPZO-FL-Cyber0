from dataclasses import dataclass
from math import sqrt

from opacus.accountants.utils import get_noise_multiplier


@dataclass(frozen=True)
class FLDPConfig:
    name: str

    # Dataset / FL setting
    total_train_samples: int
    num_clients: int
    num_sample_clients: int
    batch_size: int

    # Training setting
    rounds: int
    local_steps: int
    num_perturbations: int

    # DP setting
    clip_threshold: float
    delta: float = 1e-5

    @property
    def samples_per_client(self) -> float:
        return self.total_train_samples / self.num_clients

    @property
    def client_sample_rate(self) -> float:
        return min(1.0, self.num_sample_clients / self.num_clients)

    @property
    def batch_sample_rate_within_client(self) -> float:
        return min(1.0, self.batch_size / self.samples_per_client)

    @property
    def effective_record_sample_rate(self) -> float:
        """
        Approximate record-level sampling rate.

        A record is used in a local step when:
        1. its client is sampled;
        2. it appears in the current local batch.
        """
        return min(
            1.0,
            self.client_sample_rate * self.batch_sample_rate_within_client,
        )

    @property
    def dp_steps(self) -> int:
        """
        Current implementation applies clipping/noising once per local step.
        For one record, the approximate number of mechanisms is rounds * local_steps.
        Do not multiply by num_sample_clients here.
        """
        return self.rounds * self.local_steps

    @property
    def elementwise_vector_l2_bound(self) -> float:
        """
        Current implementation clips each scalar element independently.
        If we view P perturbation scalars as one vector, its L2 upper bound is C * sqrt(P).
        """
        return self.clip_threshold * sqrt(self.num_perturbations)


def compute_noise_multiplier(target_epsilon: float, cfg: FLDPConfig) -> float:
    return get_noise_multiplier(
        target_epsilon=target_epsilon,
        target_delta=cfg.delta,
        sample_rate=cfg.effective_record_sample_rate,
        steps=cfg.dp_steps,
        accountant="rdp",
        epsilon_tolerance=1e-4,
    )


def print_noise_table(cfg: FLDPConfig, target_epsilons=(2.0, 5.0, 8.0, 10.0)):
    print(f"\n=== {cfg.name} ===")
    print(f"total_train_samples          = {cfg.total_train_samples}")
    print(f"num_clients                  = {cfg.num_clients}")
    print(f"num_sample_clients           = {cfg.num_sample_clients}")
    print(f"samples_per_client approx    = {cfg.samples_per_client:.2f}")
    print(f"batch_size                   = {cfg.batch_size}")
    print(f"rounds                       = {cfg.rounds}")
    print(f"local_steps                  = {cfg.local_steps}")
    print(f"num_perturbations            = {cfg.num_perturbations}")
    print(f"clip_threshold C             = {cfg.clip_threshold}")
    print(f"delta                        = {cfg.delta}")
    print(f"client_sample_rate           = {cfg.client_sample_rate:.6f}")
    print(f"batch_sample_rate/client     = {cfg.batch_sample_rate_within_client:.6f}")
    print(f"effective_record_sample_rate = {cfg.effective_record_sample_rate:.6f}")
    print(f"dp_steps                     = {cfg.dp_steps}")
    print("-" * 80)

    for eps in target_epsilons:
        noise_multiplier = compute_noise_multiplier(eps, cfg)

        # 这个是当前代码最应该填入 args.dp_sigma 的值：
        # 因为 client_llm.py 里 dp_sigma 是 absolute Gaussian std。
        scalar_only_abs_sigma = noise_multiplier * cfg.clip_threshold

        # 更保守的 P 维向量口径，先不建议作为主实验。
        p_dim_l2_abs_sigma = noise_multiplier * cfg.elementwise_vector_l2_bound

        # DPZero per-sample scalar 机制的参考值。
        # 你当前代码还不是这个机制，所以这里只作为参考。
        dpzero_style_reference = (
            2.0 * noise_multiplier * cfg.clip_threshold / cfg.batch_size
        )

        print(f"target epsilon = {eps:>4.1f}")
        print(f"  noise_multiplier                    = {noise_multiplier:.6f}")
        print(f"  scalar-only abs sigma [USE THIS]    = {scalar_only_abs_sigma:.6f}")
        print(f"  P-dim L2 abs sigma [conservative]   = {p_dim_l2_abs_sigma:.6f}")
        print(f"  DPZero-style reference              = {dpzero_style_reference:.6f}")
        print()


if __name__ == "__main__":
    # 对应 decom_fl_llm_main.py 当前 SST-2 / OPT-125M LoRA 配置：
    # num_clients=20
    # num_sample_clients=20
    # rounds=100
    # local_steps=5
    # train_batch_size=8
    # zo_n_pert=10
    # C 根据 raw scalar calibration 设为 15.0
    llm_sst2_cfg = FLDPConfig(
        name="LLM SST-2 / OPT-125M LoRA / FL-ZO current mechanism",
        total_train_samples=67349,
        num_clients=20,
        num_sample_clients=20,
        batch_size=8,
        rounds=100,
        local_steps=5,
        num_perturbations=10,
        clip_threshold=15.0,
        delta=1e-5,
    )

    print_noise_table(llm_sst2_cfg)