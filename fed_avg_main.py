import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import random
import numpy as np
import pandas as pd
import os

from fed_avg.client import FedAvgClient
from fed_avg.server import FedAvgServer
from models import CNN_MNIST
from util.metrics import accuracy
from util.data_utils import get_mnist_dataloaders 

# 1.Global configuration parameters
class Args:
    num_clients = 10         # Total number of clients
    num_sample_clients = 5   # Number of clients randomly selected per round (C * K)
    rounds = 100              # Total training rounds (Communication Rounds)
    local_steps = 3          # Client-side local training steps (Local Epochs/Steps)
    lr = 0.01                # Learning rate
    batch_size = 32          # Local training Batch Size
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42                # Random seed, ensuring reproducibility
    eval_iterations = 1

args = Args()

def set_seed(seed):
    """Set a random seed to ensure reproducible results."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def setup_system():
    """Initialise the system: load data, models, clients and servers"""
    print(f"Initialising the system (Device: {args.device})...")
    
    # 1. Prepare the data (Reverted to MNIST)
    client_loaders, test_loader = get_mnist_dataloaders(
        num_clients=args.num_clients, 
        batch_size=args.batch_size
    )

    # 2. Prepare the global model (Reverted to CNN_MNIST)
    global_model = CNN_MNIST().to(args.device)
    
    # 3. Define a common inference and loss function (shared by both client and server)
    def model_inference(model, x): 
        return model(x)
    
    criterion = nn.CrossEntropyLoss()

    # 4. Initialise all clients
    clients = []
    print(f"Creating {args.num_clients} clients...")
    for i in range(args.num_clients):
        # Initialise a separate local model copy for each client (Reverted to CNN_MNIST)
        local_model = CNN_MNIST().to(args.device)
        local_model.load_state_dict(global_model.state_dict())
        
        # Initialise the optimiser
        optimizer = optim.SGD(local_model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        
        client = FedAvgClient(
            model=local_model,
            model_inference=model_inference,
            dataloader=client_loaders[i], # Allocate the corresponding data slice
            optimizer=optimizer,
            criterion=criterion,
            accuracy_func=accuracy,
            device=torch.device(args.device)
        )
        clients.append(client)

    # 5. Initialise the server
    server = FedAvgServer(
        clients=clients,
        device=torch.device(args.device),
        server_model=global_model,
        server_model_inference=model_inference,
        server_criterion=criterion,
        server_accuracy_func=accuracy,
        num_sample_clients=args.num_sample_clients,
        local_update_steps=args.local_steps
    )
    
    return server, test_loader

if __name__ == "__main__":

    set_seed(args.seed)

    server, test_loader = setup_system()
    
    # Updated print statement
    print(f"\n Start FedAvg training (CNN + MNIST)")
    print(f"   - Rounds: {args.rounds}")
    print(f"   - Clients per round: {args.num_sample_clients}")
    
    csv_filename = "fedavg_history.csv"
   
    pd.DataFrame(columns=["Round", "Train_Loss", "Test_Loss", "Test_Acc"]).to_csv(
        csv_filename, index=False, mode='w'
    )
    # =================================================

    # 3. Training cycle
    milestones = [50, 75]
    with tqdm(range(args.rounds), desc="Training Rounds") as t:
        for round_idx in t:
            
            if round_idx in milestones:
                args.lr *= 0.1  
                print(f"\n[Round {round_idx}] Decay LR to {args.lr}")
                
                for client in server.clients:
                    for param_group in client.optimizer.param_groups:
                        param_group['lr'] = args.lr
            
            train_loss, train_acc = server.train_one_step()
            
            postfix_dict = {
                "Train Loss": f"{train_loss:.4f}"
            }
            
            if args.eval_iterations != 0 and (round_idx + 1) % args.eval_iterations == 0:
                test_loss, test_acc = server.eval_model(test_loader)
                postfix_dict["Test Acc"] = f"{test_acc*100:.2f}%"
                
                row = pd.DataFrame([{
                    "Round": round_idx + 1,
                    "Train_Loss": train_loss,
                    "Test_Loss": test_loss,
                    "Test_Acc": test_acc
                }])
               
                row.to_csv(csv_filename, mode='a', header=False, index=False)
                   
            t.set_postfix(postfix_dict)

    print("\n Training complete!")
    if 'test_acc' in locals():
        print(f"Final test set accuracy: {test_acc*100:.2f}%")
        print(f"Results saved to: {csv_filename}")