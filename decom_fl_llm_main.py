"""
Legacy entry point kept for compatibility.

The LoRA-based OPT main has been replaced by the full-model RoBERTa
forward-only ZO pipeline in decom_fl_llm_roberta_main.py.

Please run:
    python decom_fl_llm_roberta_main.py
"""

from decom_fl_llm_roberta_main import *
