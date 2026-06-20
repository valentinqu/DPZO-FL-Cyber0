from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import os

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


SUPPORTED_LLM = {
    "roberta-base": "FacebookAI/roberta-base",
    "roberta-large": "FacebookAI/roberta-large",
    "opt-125m": "facebook/opt-125m",  # kept only for old exploratory scripts
}


def get_hf_tokenizer(
    hf_model_name: str,
    padding_side: str = "right",
    truncation_side: str = "right",
):
    """Load a HuggingFace tokenizer without hard-coded access tokens."""
    hf_token = os.getenv("HF_TOKEN", None)

    kwargs = {
        "padding_side": padding_side,
        "truncation_side": truncation_side,
    }
    if hf_token:
        kwargs["token"] = hf_token

    tokenizer = AutoTokenizer.from_pretrained(hf_model_name, **kwargs)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    return tokenizer


# ============================================================
# RoBERTa masked-prompt classification utilities
# ============================================================
@dataclass
class RobertaBatchInput:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    mask_positions: torch.Tensor

    def to(self, device=None, dtype=None):
        # input_ids and attention_mask must stay integer tensors.
        self.input_ids = self.input_ids.to(device=device)
        self.attention_mask = self.attention_mask.to(device=device)
        self.mask_positions = self.mask_positions.to(device=device)
        return self


@dataclass
class MaskedLMZOOutput:
    logits: torch.Tensor
    mask_positions: torch.Tensor


class RobertaSST2Template:
    """
    Prompt template for SST-2 masked-token classification.

    Example:
        sentence:  "A charming movie."
        prompt:    "A charming movie. It was <mask>."
        label:     0 -> bad, 1 -> good
    """

    verbalizer = {0: "bad", 1: "good"}

    def encode(self, sample: dict, tokenizer) -> str:
        sentence = sample["sentence"].strip()
        return f"{sentence} It was {tokenizer.mask_token}."

    def label(self, sample: dict) -> int:
        return int(sample["label"])


def _single_token_id(tokenizer, word: str) -> int:
    """
    RoBERTa uses byte-level BPE, so verbalizers should usually include a leading
    space to become one token in a natural prompt context.
    """
    candidates = [f" {word}", word]
    for candidate in candidates:
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    raise ValueError(
        f"Verbalizer word '{word}' is not a single token for tokenizer {tokenizer.name_or_path}."
    )


def get_roberta_sst2_verbalizer_ids(
    tokenizer,
    negative_word: str = "bad",
    positive_word: str = "good",
) -> list[int]:
    """Return verbalizer ids in label order: [negative, positive]."""
    return [
        _single_token_id(tokenizer, negative_word),
        _single_token_id(tokenizer, positive_word),
    ]


class RobertaSST2PromptDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length: int = 64):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template = RobertaSST2Template()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        prompt = self.template.encode(sample, self.tokenizer)
        label = self.template.label(sample)
        return {"text": prompt, "label": label}


def get_roberta_sst2_collate_fn(tokenizer, max_length: int = 64):
    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        raise ValueError("The tokenizer must define mask_token_id for RoBERTa MLM classification.")

    def collate_fn(batch: Sequence[dict]):
        texts = [item["text"] for item in batch]
        labels = torch.tensor([int(item["label"]) for item in batch], dtype=torch.long)

        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        mask_matches = input_ids.eq(mask_token_id)
        mask_counts = mask_matches.sum(dim=1)
        if not torch.all(mask_counts == 1):
            raise ValueError(
                "Each prompt must contain exactly one <mask> token. "
                f"Mask counts in current batch: {mask_counts.tolist()}"
            )
        mask_positions = mask_matches.float().argmax(dim=1).long()

        return RobertaBatchInput(input_ids, attention_mask, mask_positions), labels

    return collate_fn


def roberta_masked_inference(model, batch_inputs: RobertaBatchInput) -> MaskedLMZOOutput:
    outputs = model(
        input_ids=batch_inputs.input_ids,
        attention_mask=batch_inputs.attention_mask,
    )
    return MaskedLMZOOutput(
        logits=outputs.logits,
        mask_positions=batch_inputs.mask_positions,
    )


def masked_token_cross_entropy_loss(
    batch_pred: MaskedLMZOOutput,
    labels: torch.Tensor,
    verbalizer_id_list: Sequence[int],
):
    logits = batch_pred.logits
    mask_positions = batch_pred.mask_positions
    batch_idx = torch.arange(logits.size(0), device=logits.device)
    mask_logits = logits[batch_idx, mask_positions]
    candidate_logits = mask_logits[:, list(verbalizer_id_list)]
    return torch.nn.functional.cross_entropy(candidate_logits, labels.long())


def masked_token_accuracy(
    batch_pred: MaskedLMZOOutput,
    labels: torch.Tensor,
    verbalizer_id_list: Sequence[int],
):
    logits = batch_pred.logits
    mask_positions = batch_pred.mask_positions
    batch_idx = torch.arange(logits.size(0), device=logits.device)
    mask_logits = logits[batch_idx, mask_positions]
    candidate_logits = mask_logits[:, list(verbalizer_id_list)]
    pred = candidate_logits.argmax(dim=1)
    return pred.eq(labels.long()).float().mean().detach().cpu()


# ============================================================
# Legacy causal-LM helpers kept for old exploratory OPT scripts.
# They are no longer used by the main RoBERTa full-ZO pipeline.
# ============================================================
@dataclass
class LLMBatchInput:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor

    def to(self, device=None, dtype=None):
        self.input_ids = self.input_ids.to(device=device)
        self.attention_mask = self.attention_mask.to(device=device)
        return self


class CustomLMDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        input_text = self.texts[idx]
        input_ids = self.tokenizer.encode(input_text, add_special_tokens=True)
        if len(input_ids) > self.max_length:
            input_ids = input_ids[-self.max_length:]
        return torch.tensor(input_ids, dtype=torch.long)


class SST2Template:
    verbalizer = {0: " bad", 1: " good"}

    def get_verbalizer_id(self, tokenizer):
        return {k: tokenizer.encode(v)[-1] for k, v in self.verbalizer.items()}

    def verbalize_for_pred(self, sample):
        text = sample["sentence"].strip()
        return f"{text} It was"

    def verbalize(self, sample):
        label = sample["label"]
        return f"{self.verbalize_for_pred(sample)}{self.verbalizer[label]}"


def get_collate_fn(tokenizer, max_length):
    def collate_fn(batch):
        padded_batch = tokenizer.pad(
            {"input_ids": batch},
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = padded_batch["input_ids"]
        attention_mask = padded_batch["attention_mask"]
        return (
            LLMBatchInput(input_ids[:, :(-1)], attention_mask[:, :(-1)]),
            input_ids[:, 1:],
        )

    return collate_fn


def last_token_cross_entropy_loss(
    batch_pred, sentence_label_tokens, verbalizer_id_map, verbalizer_id_list
):
    logits = batch_pred.logits
    last_token_batch_pred = logits[:, -1, verbalizer_id_list].view(-1, len(verbalizer_id_list))
    last_token_label = (sentence_label_tokens[:, -1] == verbalizer_id_map[1]).to(int)
    return torch.nn.functional.cross_entropy(last_token_batch_pred, last_token_label)


def last_token_accuracy(batch_pred, sentence_label_tokens, verbalizer_id_map, verbalizer_id_list):
    logits = batch_pred.logits
    last_token_batch_pred = logits[:, -1, verbalizer_id_list].view(-1, len(verbalizer_id_list))
    last_token_label = (sentence_label_tokens[:, -1] == verbalizer_id_map[1]).to(int)
    pred = last_token_batch_pred.max(1, keepdim=True)[1]
    return pred.eq(last_token_label.view_as(pred)).cpu().float().mean()
