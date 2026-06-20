import torch
from transformers import AutoModelForMaskedLM


def get_roberta_mlm_model(
    model_name: str = "FacebookAI/roberta-base",
    device: str | torch.device = "cuda",
    torch_dtype: torch.dtype | None = None,
):
    """
    Load a full RoBERTa masked-language-model for pure zero-order fine-tuning.

    Important design choice:
    - No LoRA / PEFT adapters are used here.
    - All model parameters are considered trainable from the ZO perspective.
    - The implementation should use forward-only ZO updates; no backward pass is needed.

    This keeps the LLM experiment closer to the CyBeR-0 / MeZO-style setup:
    clients transmit directional scalar values, not adapter matrices or gradients.
    """
    kwargs = {}
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype

    model = AutoModelForMaskedLM.from_pretrained(model_name, **kwargs)

    # ZO does not need autograd, but the existing estimator filters parameters
    # with requires_grad=True. Therefore we keep all parameters marked trainable.
    for param in model.parameters():
        param.requires_grad = True

    model.to(device)
    model.eval()  # Disable dropout for stable finite-difference estimates.

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print("\n====== Full-model RoBERTa ZO configuration ======")
    print(f"Model: {model_name}")
    print(f"Trainable parameters for ZO: {trainable_params:,} / {total_params:,}")
    print("LoRA/PEFT: disabled")
    print("Backpropagation: disabled by design; use forward-only ZO")
    print("=================================================\n")

    return model
