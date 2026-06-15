import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, random_split
import numpy as np

from util.data_split import dirichlet_split
# 导入平移过来的大模型语言工具
from util.language_utils import get_hf_tokenizer, SST2Template, CustomLMDataset, get_collate_fn

def get_sst2_dataloaders(num_clients=10, train_batch_size=8, test_batch_size=8, iid=True, alpha=0.5, seed=42):
    """
    符合项目 Args 风格的 SST-2 联邦数据加载器
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    max_length = 32 # SST-2 影评很短，32足够，极大节省 4070 显存

    # 1. 加载 HuggingFace 数据集与分词器 (指定使用 opt-125m 的分词规则)
    dataset = load_dataset("glue", "sst2")
    tokenizer = get_hf_tokenizer("facebook/opt-125m")
    
    raw_train_dataset = dataset["train"]
    raw_test_dataset = dataset["validation"]
    
    # 2. 文本拼接魔法：把影评转换成文字接龙格式 ("xxx. It was good/bad")
    template = SST2Template()
    encoded_train_texts = list(map(template.verbalize, raw_train_dataset))
    encoded_test_texts = list(map(template.verbalize, raw_test_dataset))
    
    # 3. 转化为自定义的 PyTorch Dataset
    train_dataset = CustomLMDataset(encoded_train_texts, tokenizer, max_length=max_length)
    test_dataset = CustomLMDataset(encoded_test_texts, tokenizer, max_length=max_length)
    
    # 4. 生成全局测试集 DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        collate_fn=get_collate_fn(tokenizer, max_length),
    )

    # 5. 联邦数据切分 (IID vs Non-IID)
    if num_clients == 1:
        client_datasets = [train_dataset]
    else:
        if iid:
            print(" [LLM Data] Using IID Split for SST-2")
            total_len = len(train_dataset)
            len_per_client = total_len // num_clients
            lengths = [len_per_client] * num_clients
            for i in range(total_len % num_clients):
                lengths[i] += 1
            client_datasets = random_split(train_dataset, lengths, generator=torch.Generator().manual_seed(seed))
        else:
            print(f" [LLM Data] Using Non-IID (Dirichlet) Split with alpha={alpha}")
            # 获取原始标签用于狄利克雷发牌
            labels = list(map(lambda x: x["label"], raw_train_dataset))
            client_datasets = dirichlet_split(train_dataset, labels, num_clients, alpha, seed)

    # 6. 为每个客户端生成 DataLoader
    client_loaders = []
    for i in range(num_clients):
        loader = DataLoader(
            client_datasets[i],
            batch_size=train_batch_size,
            shuffle=True,
            collate_fn=get_collate_fn(tokenizer, max_length),
        )
        client_loaders.append(loader)
        
    return client_loaders, test_loader