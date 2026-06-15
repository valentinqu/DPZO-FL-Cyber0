from opacus.accountants import RDPAccountant

def compute_required_noise(
    target_epsilon: float,
    total_samples: int,
    batch_size: int,
    rounds: int,
    num_sample_clients: int,
    local_steps: int,
    delta: float = 1e-5,
    tolerance: float = 1e-4
) -> float:
    """
    根据给定的epsilon 反推所需的 noise_multiplier (dp_sigma)
    """
    total_steps = rounds * num_sample_clients * local_steps 
    sample_rate = batch_size / total_samples

    low_noise = 0.01
    high_noise = 100.0
    best_noise = high_noise

    for _ in range(100):
        mid_noise = (low_noise + high_noise) / 2.0
        
        accountant = RDPAccountant()
        accountant.history = [(mid_noise, sample_rate, total_steps)]
        eps = accountant.get_epsilon(delta=delta)
        
        if eps > target_epsilon:
            low_noise = mid_noise
        else:
            best_noise = mid_noise
            high_noise = mid_noise

        if abs(eps - target_epsilon) < tolerance:
            break

    return best_noise

if __name__ == "__main__":
    TOTAL_SAMPLES = 67349        # MNIST 总样本数
    BATCH_SIZE = 8             # 客户端 batch_size
    ROUNDS = 100                 # 全局训练轮次 (如果以后改跑 200 轮，记得改这里)
    NUM_SAMPLE_CLIENTS = 5      # 每轮实际参与的客户端数量
    LOCAL_STEPS = 5              # 本地迭代步数
    DELTA = 1e-5                 # DP 的 delta (一般设为 1/样本数 或更小)
    
    target_epsilons = [2.0, 5.0, 8.0, 10.0]
    
    print("=== 联邦学习 DPZO 噪声推导工具 ===")
    print(f"设定: 样本={TOTAL_SAMPLES}, Batch={BATCH_SIZE}, 轮次={ROUNDS}, 客户端={NUM_SAMPLE_CLIENTS}, 本地步数={LOCAL_STEPS}")
    print("-" * 50)
    
    for target_eps in target_epsilons:
        required_sigma = compute_required_noise(
            target_epsilon=target_eps,
            total_samples=TOTAL_SAMPLES,
            batch_size=BATCH_SIZE,
            rounds=ROUNDS,
            num_sample_clients=NUM_SAMPLE_CLIENTS,
            local_steps=LOCAL_STEPS,
            delta=DELTA
        )
        print(f"目标 Epsilon (ε) = {target_eps:>4.1f}  -->  请在代码中设定 dp_sigma = {required_sigma:.4f}")