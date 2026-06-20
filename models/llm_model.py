import torch
from transformers import AutoModelForMaskedLM


def get_roberta_mlm_model(
    model_name: str = "FacebookAI/roberta-base",
    device: str | torch.device = "cuda",
    torch_dtype: torch.dtype | None = None,
):
    """Load a full RoBERTa masked-language-model for pure zero-order fine-tuning."""
    kwargs = {}
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    model = AutoModelForMaskedLM.from_pretrained(model_name, **kwargs)
    for param in model.parameters():
        param.requires_grad = True
    model.to(device)
    model.eval()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print("\n====== Full-model RoBERTa ZO configuration ======")
    print(f"Model: {model_name}")
    print(f"Trainable parameters for ZO: {trainable_params:,} / {total_params:,}")
    print("LoRA/PEFT: disabled")
    print("Backpropagation: disabled by design; use forward-only ZO")
    print("=================================================\n")
    return model


def get_opt_lora_model(*args, **kwargs):
    raise RuntimeError(
        "LoRA has been removed from the main LLM pipeline. "
        "Use get_roberta_mlm_model() for full-model RoBERTa ZO."
    )
