import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import random
import numpy as np
import os
import pandas as pd

from decom_fl.client import ResetClient
from decom_fl.server import CeZO_Server
from models.cnn_mnist import CNN_MNIST
from util.data_utils import get_mnist_dataloaders
# from models.resnet18_femnist import ZO_ResNet18_FEMNIST 
# from util.data_utils import get_femnist_dataloaders
from util.metrics import accuracy
from util.gradient_estimators.random_gradient_estimator import (
    RandomGradientEstimator, 
    RandomGradEstimateMethod
)

def cyber0_trimmed_mean(local_grad_scalar_list, trim_ratio=0.25, nnm_k=2):
    """
    CYBER-0 核心防御：最近邻混合 (NNM) + 截断均值 (Trimmed Mean)
    专门用于对抗 Non-IID 数据异构性引发的高方差。
    """
    # 1. 把每个客户端的标量列表，转化成张量
    client_tensors = [torch.stack(client_steps) for client_steps in local_grad_scalar_list]
    stacked_scalars = torch.stack(client_tensors, dim=0)
    num_clients = stacked_scalars.shape[0]
    
    # 将高维张量展平，以便计算客户端之间的欧氏距离
    flat_scalars = stacked_scalars.view(num_clients, -1)
    
    # 计算所有客户端两两之间的距离矩阵
    dist_matrix = torch.cdist(flat_scalars, flat_scalars, p=2)
    
    # 找到每个客户端的 k 个最近邻
    _, topk_indices = torch.topk(dist_matrix, k=nnm_k + 1, largest=False, dim=1)
    
    #每个人的新梯度 = 自己与 k 个邻居的平均值
    mixed_scalars = torch.zeros_like(stacked_scalars)
    for i in range(num_clients):
        neighbors = topk_indices[i] 
        mixed_scalars[i] = stacked_scalars[neighbors].mean(dim=0)
    # 🛡️ 鲁棒聚合模块：Trimmed Mean
    trim_num = int(num_clients * trim_ratio)
    
    if trim_num > 0:
        # 现在是对混合后的梯度进行排序和截断
        sorted_scalars, _ = torch.sort(mixed_scalars, dim=0)
        trimmed_scalars = sorted_scalars[trim_num : -trim_num]
    else:
        trimmed_scalars = mixed_scalars
        
    # 最终的全局平均
    mean_tensor = trimmed_scalars.mean(dim=0)
    
    # 拆分回列表格式返回给 Server
    return [mean_tensor[i] for i in range(mean_tensor.shape[0])]

# --- 1. Sign Flipping (SF) 攻击 ---
def sign_flip_attack(local_grad_scalar_list, num_attackers=2):
    attacked_list = []
    for i, client_scalars in enumerate(local_grad_scalar_list):
        if i < num_attackers:
            # 将梯度方向翻转并放大
            poisoned_scalars = [-s * 5.0 for s in client_scalars]
            attacked_list.append(poisoned_scalars)
        else:
            attacked_list.append(client_scalars)
    return attacked_list

# --- 2. Fall of Empires (FOE) 攻击 ---
def foe_attack(local_grad_scalar_list, num_attackers=2):
    num_clients = len(local_grad_scalar_list)
    omega = 5.0 # 破坏因子
    
    honest_list = local_grad_scalar_list[num_attackers:]
    honest_tensors = torch.stack([torch.stack(client_steps) for client_steps in honest_list])
    honest_mean = honest_tensors.mean(dim=0)
    
    # FOE 恶意更新 = (1 - omega) * 诚实均值
    malicious_tensor = (1.0 - omega) * honest_mean
    malicious_steps = [malicious_tensor[i] for i in range(malicious_tensor.shape[0])]
    
    attacked_list = []
    for i in range(num_clients):
        if i < num_attackers:
            attacked_list.append(malicious_steps)
        else:
            attacked_list.append(local_grad_scalar_list[i])
    return attacked_list

# --- 3. A Little Is Enough (ALIE) 攻击 ---
def alie_attack(local_grad_scalar_list, num_attackers=2):
    num_clients = len(local_grad_scalar_list)
    z_val = 1.0 # 偏移标准差的倍数，确保攻击隐蔽且有效
    
    honest_list = local_grad_scalar_list[num_attackers:]
    honest_tensors = torch.stack([torch.stack(client_steps) for client_steps in honest_list])
    
    honest_mean = honest_tensors.mean(dim=0)
    honest_std = honest_tensors.std(dim=0)
    
    # ALIE 恶意更新 = 诚实均值 - z * 诚实标准差 (往负方向偏移)
    malicious_tensor = honest_mean - z_val * honest_std
    malicious_steps = [malicious_tensor[i] for i in range(malicious_tensor.shape[0])]
    
    attacked_list = []
    for i in range(num_clients):
        if i < num_attackers:
            attacked_list.append(malicious_steps)
        else:
            attacked_list.append(local_grad_scalar_list[i])
    return attacked_list

class Args:
    num_clients = 20
    num_sample_clients = 20
    rounds = 100
    local_steps = 5
    
    lr = 0.005
    batch_size = 128
    weight_decay = 1e-4
    
    zo_mu = 0.01
    zo_n_pert = 20
    zo_method = RandomGradEstimateMethod.rge_central  
    paramwise = True
    
    # 2. 动态调参机制 
    adjust_perturb = False
    milestones = [40, 70]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42

    # 3. 激活差分隐私 (DP) 参数
    dp_clip_threshold = 1.0  # 裁剪阈值
    dp_sigma = 0.5820        # 高斯噪声

    #目标 Epsilon (ε) =  2.0  -->  请在代码中设定 dp_sigma = 0.7731
    #目标 Epsilon (ε) =  5.0  -->  请在代码中设定 dp_sigma = 0.5820
    #目标 Epsilon (ε) =  8.0  -->  请在代码中设定 dp_sigma = 0.5107
    #目标 Epsilon (ε) = 10.0  -->  请在代码中设定 dp_sigma = 0.4808

    #4. 攻击
    num_attackers = 4
    attack_type = "ALIE"

    #5. 增加数据切分控制
    iid = False      # 设置为 False 开启异构测试
    alpha = 0.5      # Dirichlet 的偏科程度

args = Args()

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 用于实现 Label Flipping (LF) 攻击的数据加载器包裹器
class LabelFlippedLoader:
    def __init__(self, dataloader):
        self.dataloader = dataloader
    def __iter__(self):
        for data, target in self.dataloader:
            # MNIST 标签是 0-9，翻转标签：0变9，1变8...
            flipped_target = 9 - target
            yield data, flipped_target
    def __len__(self):
        return len(self.dataloader)

def setup_system():
    print(f" Initialising DeComFL (Device: {args.device}) ...")
    print(f" Current Attack Strategy: {args.attack_type} with {args.num_attackers} attackers")

    client_loaders, test_loader = get_mnist_dataloaders(
        num_clients = args.num_clients,
        batch_size = args.batch_size,
        iid = args.iid,
        alpha = args.alpha,
        seed = args.seed
    )

    # 初始化全局模型 (CNN_MNIST)
    global_model = CNN_MNIST().to(args.device)

    trainable_params = list(global_model.parameters())

    server_optimizer = optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    criterion = nn.CrossEntropyLoss()
    
    server_estimator = RandomGradientEstimator(
        parameters=trainable_params,
        mu=args.zo_mu,
        num_pert=args.zo_n_pert,
        grad_estimate_method=args.zo_method,
        device=args.device,
        normalize_perturbation=False, 
        paramwise_perturb=args.paramwise
    )

    def model_inference(model, x): return model(x)

    clients = []
    print(" Creating clients...")
    for i in range(args.num_clients):
        # 如果是 LF 攻击且该节点是恶意节点，偷换它的 DataLoader
        if args.attack_type == "LF" and i < args.num_attackers:
            current_dataloader = LabelFlippedLoader(client_loaders[i])
        else:
            current_dataloader = client_loaders[i]

        local_model = CNN_MNIST().to(args.device)
        local_model.load_state_dict(global_model.state_dict())

        local_trainable_params = list(local_model.parameters())

        local_optimizer = optim.AdamW(
            local_trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        local_estimator = RandomGradientEstimator(
            parameters=local_trainable_params,
            mu=args.zo_mu,
            num_pert=args.zo_n_pert,
            grad_estimate_method=args.zo_method,
            device=args.device,
            normalize_perturbation=False,
            paramwise_perturb=args.paramwise
        )
        
        client = ResetClient(
            model=local_model,
            model_inference=model_inference,
            dataloader=current_dataloader,
            grad_estimator=local_estimator,
            optimizer=local_optimizer,
            criterion=criterion,
            accuracy_func=accuracy,
            device=torch.device(args.device)
        )
        client.dpzero_clip_threshold = args.dp_clip_threshold
        client.dpzero_sigma = args.dp_sigma
        clients.append(client)

    server = CeZO_Server(
        clients=clients,
        device=torch.device(args.device),
        num_sample_clients=args.num_sample_clients,
        local_update_steps=args.local_steps
    )
    
    server.set_server_model_and_criterion(
        model=global_model,
        model_inference=model_inference,
        criterion=criterion,
        accuracy_func=accuracy,
        optimizer=server_optimizer,
        gradient_estimator=server_estimator
    )

    # 7. 激活鲁棒聚合功能 (Trimmed Mean)
    server.register_aggregation_func(lambda x: cyber0_trimmed_mean(x, trim_ratio=0.25, nnm_k=12 ))
    
    if args.attack_type == "SF":
        server.register_attack_func(lambda x: sign_flip_attack(x, num_attackers=args.num_attackers))
    elif args.attack_type == "FOE":
        server.register_attack_func(lambda x: foe_attack(x, num_attackers=args.num_attackers))
    elif args.attack_type == "ALIE":
        server.register_attack_func(lambda x: alie_attack(x, num_attackers=args.num_attackers))
    else:
        # 如果是 "None" 或 "LF"，服务端不进行额外的梯度篡改，正常聚合
        server.register_attack_func(lambda x: x) 
    
    return server, test_loader

if __name__ == "__main__":
    set_seed(args.seed)
    server, test_loader = setup_system()
    
    history = {"loss": [], "acc": []}
    
    print(f"\n Start training (Total Rounds: {args.rounds})")
    print(f"   Strategy: Initial LR={args.lr}, Perturbs={args.zo_n_pert}")
    if args.adjust_perturb:
        print(f"   Dynamic adjustment point: Rounds {args.milestones}")

    with torch.no_grad(), tqdm(range(args.rounds), desc="Training") as t:
        for round_idx in t:
            train_loss, train_acc = server.train_one_step(iteration=round_idx)
            
            if args.adjust_perturb:
                if round_idx in args.milestones:
                    new_lr = server.optim.param_groups[0]["lr"] * 0.5
                    new_pert = int(server.clients[0].grad_estimator.num_pert * 2) 
                    
                    server.set_learning_rate(new_lr)
                    server.set_perturbation(new_pert)
                    
                    t.write(f"\n[Round {round_idx}] adjust parameters: LR -> {new_lr:.6f}, Perturbs -> {new_pert}")

            test_loss, test_acc = server.eval_model(test_loader)
            
            history["loss"].append(test_loss)
            history["acc"].append(test_acc)
            
            t.set_postfix({
                "Train Loss": f"{train_loss:.4f}",
                "Test Acc": f"{test_acc*100:.2f}%"
            })

            row = pd.DataFrame([{
                "loss": test_loss,
                "acc": test_acc
            }])

            row.to_csv(
                "cnn_epsilon5.0_ALIE_attacker4_a0.5_nnm.csv",
                mode="a",
                header=False,
                index=False
            )

    print(f"\n Training complete! Final accuracy: {history['acc'][-1]*100:.2f}%")