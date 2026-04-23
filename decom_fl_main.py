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

def cyber0_trimmed_mean(local_grad_scalar_list, trim_ratio=0.2):
    """
    CYBER-0 Trimmed Mean：处理嵌套列表并无缝对接 Server
    """
    # 1. 把每个客户端的 local_steps 列表，转化成张量
    client_tensors = [torch.stack(client_steps) for client_steps in local_grad_scalar_list]
    
    # 2.
    stacked_scalars = torch.stack(client_tensors, dim=0)
    
    num_clients = stacked_scalars.shape[0]
    trim_num = int(num_clients * trim_ratio)
    
    if trim_num > 0:
        # 3. 沿着客户端维度进行排序
        sorted_scalars, _ = torch.sort(stacked_scalars, dim=0)
        # 4. 剔除那 20% 的拜占庭节点
        trimmed_scalars = sorted_scalars[trim_num : -trim_num, :, :]
    else:
        trimmed_scalars = stacked_scalars
        
    # 5. 形状为 [local_steps, num_pert]
    mean_tensor = trimmed_scalars.mean(dim=0)
    
    # 6. 把张量重新拆分成列表返回
    return [mean_tensor[i] for i in range(mean_tensor.shape[0])]

def sign_flip_attack(local_grad_scalar_list):
    num_attackers = 1  
    attacked_list = []
    for i, client_scalars in enumerate(local_grad_scalar_list):
        if i < num_attackers:
            poisoned_scalars = [-s * 5.0 for s in client_scalars]
            attacked_list.append(poisoned_scalars)
        else:
            attacked_list.append(client_scalars)
    return attacked_list

class Args:
    num_clients = 10
    num_sample_clients = 5
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
    adjust_perturb = True
    milestones = [40, 70]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42

    # 3. 激活差分隐私 (DP) 参数
    dp_clip_threshold = 1.0  # 裁剪阈值
    dp_sigma = 0.1           # 高斯噪声

args = Args()

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def setup_system():
    print(f" Initialising DeComFL (Device: {args.device}) ...")

    # 4. 加载 MNIST 数据集
    client_loaders, test_loader = get_mnist_dataloaders(
        num_clients=args.num_clients, batch_size=args.batch_size
    )

    # 5. 初始化全局模型 (CNN_MNIST)
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
        # 6. 本地模型初始化 (CNN_MNIST)
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
            dataloader=client_loaders[i],
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
    server.register_attack_func(sign_flip_attack)
    server.register_aggregation_func(lambda x: cyber0_trimmed_mean(x, trim_ratio=0.2))
    
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
                "history_forward_6000.csv",
                mode="a",
                header=False,
                index=False
            )

    print(f"\n Training complete! Final accuracy: {history['acc'][-1]*100:.2f}%")