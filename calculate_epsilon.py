from dataclasses import dataclass

from opacus.accountants import RDPAccountant


@dataclass(frozen=True)
class ClientSideScalarDPConfig:
    name: str

    # Scalar mechanism
    clip_threshold: float
    delta: float = 1e-5

    # Composition setting
    # steps=1: per-upload scalar message DP
    # steps=rounds * local_steps * num_perturbations:
    #        composed privacy over the whole training transcript
    steps: int = 1

    @property
    def sensitivity(self) -> float:
        # scalar is clipped to [-C, C], so neighboring scalar messages differ by at most 2C
        return 2.0 * self.clip_threshold


def epsilon_from_abs_sigma(
    abs_sigma: float,
    cfg: ClientSideScalarDPConfig,
) -> float:
    """
    Convert absolute scalar noise std into epsilon.

    Your code uses:
        noisy_scalar = clipped_scalar + Normal(0, abs_sigma^2)

    Opacus expects:
        noise_multiplier = abs_sigma / sensitivity
    """
    if abs_sigma <= 0:
        return float("inf")

    noise_multiplier = abs_sigma / cfg.sensitivity

    accountant = RDPAccountant()

    # No sample-level subsampling here.
    # This is a scalar Gaussian mechanism on one uploaded message.
    sample_rate = 1.0

    for _ in range(cfg.steps):
        accountant.step(
            noise_multiplier=noise_multiplier,
            sample_rate=sample_rate,
        )

    return accountant.get_epsilon(delta=cfg.delta)


def sigma_for_target_epsilon(
    target_epsilon: float,
    cfg: ClientSideScalarDPConfig,
    tolerance: float = 1e-4,
) -> float:
    """
    Binary search absolute scalar noise std for a target epsilon.
    """
    low = 1e-8
    high = 1.0

    while epsilon_from_abs_sigma(high, cfg) > target_epsilon:
        high *= 2.0

    best = high

    for _ in range(100):
        mid = (low + high) / 2.0
        eps = epsilon_from_abs_sigma(mid, cfg)

        if eps > target_epsilon:
            low = mid
        else:
            best = mid
            high = mid

        if abs(eps - target_epsilon) < tolerance:
            break

    return best


def print_sigma_sweep(cfg: ClientSideScalarDPConfig, sigmas=(10.0, 20.0, 50.0, 100.0)):
    print(f"\n=== {cfg.name} ===")
    print(f"clip_threshold C = {cfg.clip_threshold}")
    print(f"sensitivity Δ    = {cfg.sensitivity}")
    print(f"delta            = {cfg.delta}")
    print(f"composition steps = {cfg.steps}")
    print("-" * 70)

    print("Epsilon for chosen absolute dp_sigma values:")
    for sigma in sigmas:
        eps = epsilon_from_abs_sigma(sigma, cfg)
        noise_multiplier = sigma / cfg.sensitivity
        print(
            f"  dp_sigma={sigma:>8.3f} | "
            f"noise_multiplier={noise_multiplier:>8.5f} | "
            f"epsilon={eps:>10.4f}"
        )
    print()


def print_target_epsilon_table(cfg: ClientSideScalarDPConfig, epsilons=(2.0, 5.0, 10.0)):
    print("Required absolute dp_sigma for target epsilons:")
    for eps in epsilons:
        sigma = sigma_for_target_epsilon(eps, cfg)
        checked_eps = epsilon_from_abs_sigma(sigma, cfg)
        print(
            f"  target epsilon={eps:>4.1f} -> "
            f"set args.dp_sigma={sigma:>10.4f} "
            f"(checked epsilon={checked_eps:.4f})"
        )
    print()


if __name__ == "__main__":
    # Current RoBERTa full-ZO client-side scalar setting
    C = 100.0
    DELTA = 1e-5

    # 1) Per-upload scalar message DP
    per_upload_cfg = ClientSideScalarDPConfig(
        name="Client-side scalar DP, per uploaded scalar",
        clip_threshold=C,
        delta=DELTA,
        steps=1,
    )

    print_sigma_sweep(per_upload_cfg, sigmas=(10.0, 20.0, 50.0, 100.0))
    print_target_epsilon_table(per_upload_cfg, epsilons=(2.0, 5.0, 10.0))

    # 2) Optional: composed over 200 uploaded scalars per client
    # Use this only if you want to report whole-transcript privacy.
    composed_cfg = ClientSideScalarDPConfig(
        name="Client-side scalar DP, composed over 200 rounds",
        clip_threshold=C,
        delta=DELTA,
        steps=200,
    )

    print_sigma_sweep(composed_cfg, sigmas=(10.0, 20.0, 50.0, 100.0))
    print_target_epsilon_table(composed_cfg, epsilons=(2.0, 5.0, 10.0))