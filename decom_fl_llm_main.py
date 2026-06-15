import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import random
import numpy as np
import pandas as pd

from decom_fl.client_llm import ResetClient
from decom_fl.server import CeZO_Server
from util.llm_data_utils import get_sst2_dataloaders
from models.llm_model import get_opt_lora_model
from util.language_utils import SST2Template, get_hf_tokenizer, last_token_cross_entropy_loss, last_token_accuracy
from util.gradient_estimators.random_gradient_estimator import RandomGradientEstimator, RandomGradEstimateMethod

# ==========================================
# 1. 核心防御：NNM + Trimmed Mean
# ==========================================
def cyber0_trimmed_mean(local_grad_scalar_list, trim_ratio=0.25, nnm_k=12):
    client_tensors = [torch.stack(client_steps) for client_steps in local_grad_scalar_list]
    stacked_scalars = torch.stack(client_tensors, dim=0)
    num_clients = stacked_scalars.shape[0]
    
    # NNM 距离混合
    flat_scalars = stacked_scalars.view(num_clients, -1)
    dist_matrix = torch.cdist(flat_scalars, flat_scalars, p=2)
    _, topk_indices = torch.topk(dist_matrix, k=nnm_k + 1, largest=False, dim=1)
    
    mixed_scalars = torch.zeros_like(stacked_scalars)
    for i in range(num_clients):
        neighbors = topk_indices[i] 
        mixed_scalars[i] = stacked_scalars[neighbors].mean(dim=0)
        
    # Trimmed Mean 截断
    trim_num = int(num_clients * trim_ratio)
    if trim_num > 0:
        sorted_scalars, _ = torch.sort(mixed_scalars, dim=0)
        trimmed_scalars = sorted_scalars[trim_num : -trim_num]
    else:
        trimmed_scalars = mixed_scalars
        
    mean_tensor = trimmed_scalars.mean(dim=0)
    return [mean_tensor[i] for i in range(mean_tensor.shape[0])]

# ==========================================
# 2. 攻击者：FOE 攻击
# ==========================================
def foe_attack(local_grad_scalar_list, num_attackers=4):
    num_clients = len(local_grad_scalar_list)
    omega = 5.0 
    honest_list = local_grad_scalar_list[num_attackers:]
    honest_tensors = torch.stack([torch.stack(client_steps) for client_steps in honest_list])
    honest_mean = honest_tensors.mean(dim=0)
    
    malicious_tensor = (1.0 - omega) * honest_mean
    malicious_steps = [malicious_tensor[i] for i in range(malicious_tensor.shape[0])]
    
    attacked_list = []
    for i in range(num_clients):
        if i < num_attackers:
            attacked_list.append(malicious_steps)
        else:
            attacked_list.append(local_grad_scalar_list[i])
    return attacked_list

# ==========================================
# 3. 实验超参数配置
# ==========================================
class Args:
    num_clients = 20
    num_sample_clients = 20
    rounds = 100
    local_steps = 5
    
    lr = 5e-4               # LLM 的学习率
    train_batch_size = 8
    test_batch_size = 16
    weight_decay = 0.0
    
    zo_mu = 0.001
    zo_n_pert = 10
    zo_method = RandomGradEstimateMethod.rge_central  
    paramwise = False
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42

    # DP 与 攻击配置
    dp_clip_threshold = 1000.0  #33.7249
    dp_sigma = 0.0

    #目标 Epsilon (ε) =  2.0  -->  请在代码中设定 dp_sigma = 0.5566
    #目标 Epsilon (ε) =  5.0  -->  请在代码中设定 dp_sigma = 0.4028
    #目标 Epsilon (ε) =  8.0  -->  请在代码中设定 dp_sigma = 0.3452
    #目标 Epsilon (ε) = 10.0  -->  请在代码中设定 dp_sigma = 0.3220

    num_attackers = 4
    attack_type = "FOE"

    # 数据异构配置
    iid = True     
    alpha = 0.5      

args = Args()

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ==========================================
# 4. 系统搭建与封装
# ==========================================
def setup_system():
    print(f" [Init] LLM DeComFL Starting on {args.device}...")
    
    # 4.1 获取大语言模型数据 (SST-2)
    client_loaders, test_loader = get_sst2_dataloaders(
        num_clients=args.num_clients,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
        iid=args.iid,
        alpha=args.alpha,
        seed=args.seed
    )

    # 4.2 加载全局大模型 (挂载 LoRA)
    print(" [Init] Loading Shared Global OPT-125M with LoRA...")

    global_model = get_opt_lora_model(device=args.device)
    
    # parameters  requires_grad=True 的 LoRA 参数
    trainable_params = [p for p in global_model.parameters() if p.requires_grad]

    server_optimizer = optim.SGD(trainable_params, lr=args.lr) # LLM 联邦聚合常用 SGD
    
    server_estimator = RandomGradientEstimator(
        parameters=trainable_params,
        mu=args.zo_mu,
        num_pert=args.zo_n_pert,
        grad_estimate_method=args.zo_method,
        device=args.device,
        normalize_perturbation=False, 
        paramwise_perturb=args.paramwise
    )

    # 4.3  LLM 的 Inference 和 Loss 
    tokenizer = get_hf_tokenizer("facebook/opt-125m")
    template = SST2Template()
    verbalizer_id_map = template.get_verbalizer_id(tokenizer)
    verbalizer_id_list = [verbalizer_id_map[i] for i in range(len(verbalizer_id_map))]

    def llm_inference(model, x):

        return model(input_ids=x.input_ids.to(args.device), attention_mask=x.attention_mask.to(args.device))

    def llm_criterion(outputs, targets):
        return last_token_cross_entropy_loss(outputs, targets.to(args.device), verbalizer_id_map, verbalizer_id_list)
        
    def llm_accuracy(outputs, targets):
        return last_token_accuracy(outputs, targets.to(args.device), verbalizer_id_map, verbalizer_id_list)

    # 4.4 初始化客户端群
    clients = []
    print(f" [Init] Creating {args.num_clients} Clients (Time-Multiplexing Shared Model)...")
    for i in range(args.num_clients):
        

        local_trainable_params = [p for p in global_model.parameters() if p.requires_grad]

        local_optimizer = optim.SGD(local_trainable_params, lr=args.lr)
        
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
            model=global_model,               
            model_inference=llm_inference,    
            dataloader=client_loaders[i],
            grad_estimator=local_estimator,
            optimizer=local_optimizer,
            criterion=llm_criterion,          
            accuracy_func=llm_accuracy,       
            device=torch.device(args.device)
        )
        client.dpzero_clip_threshold = args.dp_clip_threshold
        client.dpzero_sigma = args.dp_sigma
        clients.append(client)

    # 4.5 初始化服务端
    server = CeZO_Server(
        clients=clients,
        device=torch.device(args.device),
        num_sample_clients=args.num_sample_clients,
        local_update_steps=args.local_steps
    )
    
    server.set_server_model_and_criterion(
        model=global_model,
        model_inference=llm_inference,
        criterion=llm_criterion,
        accuracy_func=llm_accuracy,
        optimizer=server_optimizer,
        gradient_estimator=server_estimator
    )

    # 4.6 注册防御与攻击
    server.register_aggregation_func(lambda x: cyber0_trimmed_mean(x, trim_ratio=0.20, nnm_k=0))
    
    if args.attack_type == "FOE":
        server.register_attack_func(lambda x: foe_attack(x, num_attackers=args.num_attackers))
    else:
        server.register_attack_func(lambda x: x) 
    
    return server, test_loader

# ==========================================
# 5. 训练循环
# ==========================================
if __name__ == "__main__":
    set_seed(args.seed)
    server, test_loader = setup_system()
    
    history = {"loss": [], "acc": []}
    
    print(f"\n [Training] Starts (Total Rounds: {args.rounds})")

    with torch.no_grad(), tqdm(range(args.rounds), desc="Training") as t:
        for round_idx in t:
            # 下发模型 -> 本地更新 -> 攻击 -> 防御 -> 聚合
            train_loss, train_acc = server.train_one_step(iteration=round_idx)
            
            # 全局评估
            test_loss, test_acc = server.eval_model(test_loader)
            
            history["loss"].append(test_loss)
            history["acc"].append(test_acc)
            
            t.set_postfix({
                "Test Loss": f"{test_loss:.4f}",
                "Test Acc": f"{test_acc*100:.2f}%"
            })

            # 保存数据到 CSV
            row = pd.DataFrame([{"loss": test_loss, "acc": test_acc}])
            row.to_csv("llm_sst2_iid_nodp_attack_nodefense_20clients.csv", mode="a", header=False, index=False)

    print(f"\n Training complete! Final LLM Accuracy: {history['acc'][-1]*100:.2f}%")