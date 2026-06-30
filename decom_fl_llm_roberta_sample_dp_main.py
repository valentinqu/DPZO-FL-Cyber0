import os
import random
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from tqdm import tqdm

from decom_fl.client_llm_sample_dp import SampleLevelDPClient
from decom_fl.server import CeZO_Server
from models.llm_model import get_roberta_mlm_model
from util.llm_data_utils import get_sst2_dataloaders
from util.language_utils import (
    get_hf_tokenizer,
    get_roberta_sst2_verbalizer_ids,
    roberta_masked_inference,
    masked_token_cross_entropy_loss,
    masked_token_accuracy,
)
from util.gradient_estimators.random_gradient_estimator import (
    RandomGradientEstimator,
    RandomGradEstimateMethod,
)
from util.gradient_estimators.sample_level_dp_random_gradient_estimator import (
    SampleLevelDPRandomGradientEstimator,
)


# ==========================================
# Robust aggregation and attack: same server interface as the main pipeline
# ==========================================
def cyber0_trimmed_mean(local_grad_scalar_list, trim_ratio=0.25, nnm_k=0):
    client_tensors = [torch.stack(client_steps) for client_steps in local_grad_scalar_list]
    stacked_scalars = torch.stack(client_tensors, dim=0)
    num_clients = stacked_scalars.shape[0]

    if nnm_k > 0:
        flat_scalars = stacked_scalars.view(num_clients, -1)
        dist_matrix = torch.cdist(flat_scalars, flat_scalars, p=2)
        _, topk_indices = torch.topk(dist_matrix, k=nnm_k + 1, largest=False, dim=1)
        mixed_scalars = torch.zeros_like(stacked_scalars)
        for i in range(num_clients):
            mixed_scalars[i] = stacked_scalars[topk_indices[i]].mean(dim=0)
    else:
        mixed_scalars = stacked_scalars

    trim_num = int(num_clients * trim_ratio)
    if trim_num > 0:
        sorted_scalars, _ = torch.sort(mixed_scalars, dim=0)
        trimmed_scalars = sorted_scalars[trim_num:-trim_num]
    else:
        trimmed_scalars = mixed_scalars

    mean_tensor = trimmed_scalars.mean(dim=0)
    return [mean_tensor[i] for i in range(mean_tensor.shape[0])]


def foe_attack(local_grad_scalar_list, num_attackers=2, omega=5.0):
    num_clients = len(local_grad_scalar_list)
    if num_attackers <= 0:
        return local_grad_scalar_list
    if num_attackers >= num_clients:
        raise ValueError("num_attackers must be smaller than sampled clients")

    honest_list = local_grad_scalar_list[num_attackers:]
    honest_tensors = torch.stack([torch.stack(client_steps) for client_steps in honest_list])
    honest_mean = honest_tensors.mean(dim=0)

    malicious_tensor = (1.0 - omega) * honest_mean
    malicious_steps = [malicious_tensor[i] for i in range(malicious_tensor.shape[0])]

    attacked = []
    for i in range(num_clients):
        attacked.append(malicious_steps if i < num_attackers else local_grad_scalar_list[i])
    return attacked


# ==========================================
# Experiment configuration
# ==========================================
class Args:
    # Model / data
    model_name = "FacebookAI/roberta-base"
    dataset_name = "sst2"
    max_length = 64
    max_train_samples = 512
    max_test_samples = None

    # Federated setting
    num_clients = 8
    num_sample_clients = 8
    num_attackers = 2
    iid = True
    alpha = 0.5

    # Training / ZO
    rounds = 100
    local_steps = 1
    train_batch_size = 16
    test_batch_size = 32
    eval_interval = 10
    lr = 1e-6
    weight_decay = 0.0

    zo_mu = 1e-3
    zo_n_pert = 1
    zo_method = RandomGradEstimateMethod.rge_central
    paramwise = True
    sgd_only_no_optim = True

    # Sample-level DPZero-style scalar DP.
    # sample_dp_sigma is absolute std added to the averaged clipped per-sample scalar.
    sample_dp_clip_threshold = 100.0
    sample_dp_sigma = 0.0

    # Example values from an Opacus sample-level accountant should be converted
    # to this absolute averaged-scalar sigma before being used here.

    # Attack / defense
    attack_type = "None"  # "None" or "FOE"
    foe_omega = 5.0
    trim_ratio = 0.0
    nnm_k = 0

    # Misc
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    output_csv = "roberta_sst2_sampledp_clean.csv"
    overwrite_output = True


args = Args()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_output_file():
    if args.overwrite_output and os.path.exists(args.output_csv):
        os.remove(args.output_csv)
    output_dir = os.path.dirname(args.output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)


def setup_system():
    client_loaders, test_loader = get_sst2_dataloaders(
        num_clients=args.num_clients,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
        iid=args.iid,
        alpha=args.alpha,
        seed=args.seed,
        model_name=args.model_name,
        max_length=args.max_length,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
    )

    global_model = get_roberta_mlm_model(model_name=args.model_name, device=args.device)
    trainable_params = [p for p in global_model.parameters() if p.requires_grad]
    server_optimizer = optim.SGD(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    server_estimator = RandomGradientEstimator(
        parameters=trainable_params,
        mu=args.zo_mu,
        num_pert=args.zo_n_pert,
        grad_estimate_method=args.zo_method,
        device=args.device,
        normalize_perturbation=False,
        paramwise_perturb=args.paramwise,
        sgd_only_no_optim=args.sgd_only_no_optim,
    )

    tokenizer = get_hf_tokenizer(args.model_name, padding_side="right", truncation_side="left")
    verbalizer_id_list = get_roberta_sst2_verbalizer_ids(tokenizer)

    def llm_inference(model, x):
        return roberta_masked_inference(model, x)

    def llm_criterion(outputs, targets):
        return masked_token_cross_entropy_loss(outputs, targets, verbalizer_id_list)

    def llm_accuracy(outputs, targets):
        return masked_token_accuracy(outputs, targets, verbalizer_id_list)

    def llm_per_sample_criterion(outputs, targets):
        logits = outputs.logits
        mask_positions = outputs.mask_positions
        batch_idx = torch.arange(logits.size(0), device=logits.device)
        mask_logits = logits[batch_idx, mask_positions]
        candidate_logits = mask_logits[:, list(verbalizer_id_list)]
        return torch.nn.functional.cross_entropy(
            candidate_logits,
            targets.long(),
            reduction="none",
        )

    clients = []
    for i in range(args.num_clients):
        local_estimator = SampleLevelDPRandomGradientEstimator(
            parameters=trainable_params,
            mu=args.zo_mu,
            num_pert=args.zo_n_pert,
            grad_estimate_method=args.zo_method,
            device=args.device,
            normalize_perturbation=False,
            paramwise_perturb=args.paramwise,
            sgd_only_no_optim=args.sgd_only_no_optim,
            sample_dp_clip_threshold=args.sample_dp_clip_threshold,
            sample_dp_sigma=args.sample_dp_sigma,
        )
        client = SampleLevelDPClient(
            model=global_model,
            model_inference=llm_inference,
            dataloader=client_loaders[i],
            grad_estimator=local_estimator,
            criterion=llm_criterion,
            per_sample_criterion=llm_per_sample_criterion,
            accuracy_func=llm_accuracy,
            device=torch.device(args.device),
            client_id=i,
        )
        clients.append(client)

    server = CeZO_Server(
        clients=clients,
        device=torch.device(args.device),
        num_sample_clients=args.num_sample_clients,
        local_update_steps=args.local_steps,
    )
    server.set_server_model_and_criterion(
        model=global_model,
        model_inference=llm_inference,
        criterion=llm_criterion,
        accuracy_func=llm_accuracy,
        optimizer=server_optimizer,
        gradient_estimator=server_estimator,
    )

    server.register_aggregation_func(lambda x: cyber0_trimmed_mean(x, args.trim_ratio, args.nnm_k))
    if args.attack_type == "FOE":
        server.register_attack_func(lambda x: foe_attack(x, args.num_attackers, args.foe_omega))
    else:
        server.register_attack_func(lambda x: x)

    return server, test_loader


if __name__ == "__main__":
    set_seed(args.seed)
    prepare_output_file()
    server, test_loader = setup_system()
    history = {"loss": [], "acc": []}

    with torch.no_grad(), tqdm(range(args.rounds), desc="Sample-level DP ZO") as t:
        for round_idx in t:
            train_loss, train_acc = server.train_one_step(iteration=round_idx)
            if round_idx == 0 or (round_idx + 1) % args.eval_interval == 0:
                test_loss, test_acc = server.eval_model(test_loader)
            else:
                test_loss, test_acc = history["loss"][-1], history["acc"][-1]

            history["loss"].append(test_loss)
            history["acc"].append(test_acc)
            t.set_postfix({
                "Train Loss": f"{train_loss:.4f}",
                "Test Loss": f"{test_loss:.4f}",
                "Test Acc": f"{test_acc * 100:.2f}%",
            })
            pd.DataFrame([{"loss": test_loss, "acc": test_acc}]).to_csv(
                args.output_csv,
                mode="a",
                header=False,
                index=False,
            )

    print(f"Final LLM Accuracy: {history['acc'][-1] * 100:.2f}%")
