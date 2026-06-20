"""Compatibility wrapper for older imports.

The main LLM pipeline now uses full-model RoBERTa forward-only ZO.
LoRA/PEFT is intentionally disabled.
"""

from models.llm_model import get_roberta_mlm_model

__all__ = ["get_roberta_mlm_model"]
