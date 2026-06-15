import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

def get_opt_lora_model(device="cuda"):
    """
    加载由 Google/Facebook 论文推荐的 opt-125m 并植入 LoRA 适配器
    """
    hf_model_name = "facebook/opt-125m"
    
    # 1. 以因果语言模型（Causal LM 文字接龙）形式加载预训练权重
    model = AutoModelForCausalLM.from_pretrained(hf_model_name)
    
    # 2. 配置低秩矩阵 LoRA 
    lora_config = LoraConfig(
        r=8,                # 秩大小
        lora_alpha=16,      # 缩放系数
        target_modules=["q_proj", "v_proj"], # 仅让 Attention 的 Q 和 V 矩阵参与联邦训练
    )
    
    # 3. 挂载 LoRA
    lora_model = get_peft_model(model, lora_config)
    
    # 打印可训练参数占比
    print("\n====== LoRA 参数配置成功 ======")
    lora_model.print_trainable_parameters()
    print("================================\n")
    
    return lora_model.to(device)