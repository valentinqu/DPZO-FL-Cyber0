import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, random_split
import numpy as np

from util.data_split import dirichlet_split
from util.language_utils import (
    get_hf_tokenizer,
    RobertaSST2PromptDataset,
    get_roberta_sst2_collate_fn,
)


def _maybe_select_subset(hf_dataset, max_samples: int | None, seed: int):
    if max_samples is None or max_samples <= 0 or max_samples >= len(hf_dataset):
        return hf_dataset
    return hf_dataset.shuffle(seed=seed).select(range(max_samples))


def get_sst2_dataloaders(
    num_clients: int = 8,
    train_batch_size: int = 16,
    test_batch_size: int = 32,
    iid: bool = True,
    alpha: float = 0.5,
    seed: int = 42,
    model_name: str = "FacebookAI/roberta-base",
    max_length: int = 64,
    max_train_samples: int | None = 512,
    max_test_samples: int | None = None,
):
    """
    SST-2 federated dataloaders for the full-model RoBERTa ZO pipeline.

    This loader follows a prompt-based masked-LM setting closer to CyBeR-0:
        sentence + " It was <mask>."

    Notes:
    - No LoRA/PEFT assumption is made here.
    - max_train_samples defaults to 512 to match the paper-like few-shot setting.
    - If max_train_samples=None, the full SST-2 train split is used.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    raw_dataset = load_dataset("glue", "sst2")
    tokenizer = get_hf_tokenizer(
        model_name,
        padding_side="right",
        truncation_side="left",
    )

    raw_train_dataset = _maybe_select_subset(raw_dataset["train"], max_train_samples, seed)
    raw_test_dataset = _maybe_select_subset(raw_dataset["validation"], max_test_samples, seed)

    train_dataset = RobertaSST2PromptDataset(
        raw_train_dataset,
        tokenizer,
        max_length=max_length,
    )
    test_dataset = RobertaSST2PromptDataset(
        raw_test_dataset,
        tokenizer,
        max_length=max_length,
    )

    collate_fn = get_roberta_sst2_collate_fn(tokenizer, max_length=max_length)

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

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
            client_datasets = random_split(
                train_dataset,
                lengths,
                generator=torch.Generator().manual_seed(seed),
            )
        else:
            print(f" [LLM Data] Using Non-IID (Dirichlet) Split with alpha={alpha}")
            labels = [int(raw_train_dataset[i]["label"]) for i in range(len(raw_train_dataset))]
            client_datasets = dirichlet_split(
                train_dataset,
                labels,
                num_clients,
                alpha,
                seed,
            )

    client_loaders = []
    for i in range(num_clients):
        loader = DataLoader(
            client_datasets[i],
            batch_size=train_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        client_loaders.append(loader)

    print(
        " [LLM Data] SST-2 ready: "
        f"train={len(train_dataset)}, test={len(test_dataset)}, "
        f"clients={num_clients}, batch={train_batch_size}, model={model_name}"
    )

    return client_loaders, test_loader
