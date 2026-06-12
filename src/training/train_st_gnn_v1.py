import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import time
import json

from src.models.stgnn_v1.st_gnn_model import UHSGraphDataset, ST_GNN

def evaluate_model(model, dataset, device, target_dp_std, target_dsg_std):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for i in range(len(dataset)):
            x, target, _ = dataset.get_scaled_features(i)
            x = x.to(device)
            target = target.to(device)
            
            # Normalize target
            target_norm = torch.zeros_like(target)
            target_norm[:, 0] = target[:, 0] / target_dp_std
            target_norm[:, 1] = target[:, 1] / target_dsg_std
            
            pred_norm = model(x, dataset.edge_index.to(device), dataset.edge_trans.to(device))
            loss = F_mse(pred_norm, target_norm)
            total_loss += loss.item()
            
    return total_loss / len(dataset)

def F_mse(pred, target):
    return torch.mean((pred - target)**2)

def main():
    print("=" * 80)
    print("PHASE 5: TRAINING ST-GNN SURROGATE MODEL")
    print("=" * 80)
    
    project_root = Path(__file__).resolve().parents[2]
    data_raw_dir = project_root / "data" / "raw"
    data_processed_dir = project_root / "data" / "processed"
    checkpoint_dir = project_root / "checkpoints" / "stgnn_v1"
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    data_processed_dir.mkdir(parents=True, exist_ok=True)
    
    graph_path = data_processed_dir / "mesh_graph.npz"
    
    if not graph_path.exists():
        raise FileNotFoundError("Run graph_construction.py first to build the graph!")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Training Dataset (Cases 1-4)
    print("\nLoading Training Dataset (Cases 1-4)...")
    train_dataset = UHSGraphDataset(data_raw_dir, case_ids=[1, 2, 3, 4], graph_path=graph_path, stride=4)
    
    # 2. Load Validation Dataset (Case 5) using train scaler
    print("\nLoading Validation Dataset (Case 5)...")
    val_dataset = UHSGraphDataset(data_raw_dir, case_ids=[5], graph_path=graph_path, scaler=train_dataset.scaler)
    
    # 3. Calculate target standard deviations for loss scaling
    print("\nCalculating target scale statistics on training set...")
    all_targets = np.concatenate([s['target'] for s in train_dataset.samples], axis=0)
    target_dp_std = float(np.std(all_targets[:, 0]))
    target_dsg_std = float(np.std(all_targets[:, 1]))
    if target_dp_std < 1e-6: target_dp_std = 1.0
    if target_dsg_std < 1e-6: target_dsg_std = 1.0
    
    print(f"  dP target standard deviation: {target_dp_std:.4f} bar")
    print(f"  dSg target standard deviation: {target_dsg_std:.6f}")
    
    # Save statistics for inference and evaluation denormalization
    stats = {
        'feature_mean': train_dataset.scaler['mean'].tolist(),
        'feature_std': train_dataset.scaler['std'].tolist(),
        'target_dp_std': target_dp_std,
        'target_dsg_std': target_dsg_std
    }
    with open(data_processed_dir / 'st_gnn_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print("  Saved scaler statistics to st_gnn_stats.json")
    
    # 4. Instantiate Model
    # Input has 20 features from UHSGraphDataset
    in_features = train_dataset.scaler['mean'].shape[0]
    model = ST_GNN(in_features=in_features, hidden_features=64, out_features=2)
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    # 5. Training Loop
    epochs = 12
    batch_size = 16
    n_samples = len(train_dataset)
    indices = np.arange(n_samples)
    
    print(f"\nTraining on {n_samples} graph sequences (10,116 nodes each) for {epochs} epochs...")
    
    best_val_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        model.train()
        np.random.shuffle(indices)
        
        t0 = time.time()
        epoch_loss = 0.0
        
        # Mini-batch accumulation
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            optimizer.zero_grad()
            batch_loss = 0.0
            
            # Since graph sizes are 10k, we compile gradients for each graph in the batch
            for idx in batch_indices:
                x, target, _ = train_dataset.get_scaled_features(idx)
                x = x.to(device)
                target = target.to(device)
                
                # Normalize targets
                target_norm = torch.zeros_like(target)
                target_norm[:, 0] = target[:, 0] / target_dp_std
                target_norm[:, 1] = target[:, 1] / target_dsg_std
                
                # Forward
                pred_norm = model(x, train_dataset.edge_index.to(device), train_dataset.edge_trans.to(device))
                loss = F_mse(pred_norm, target_norm)
                
                # Backward step accumulation
                loss.backward()
                batch_loss += loss.item()
                
            # Perform optimization step
            optimizer.step()
            epoch_loss += batch_loss
            
        epoch_loss /= n_samples
        
        # Compute validation loss
        val_loss = evaluate_model(model, val_dataset, device, target_dp_std, target_dsg_std)
        t1 = time.time()
        
        print(f"  Epoch {epoch:2d}/{epochs:2d} | Train Loss (norm MSE) = {epoch_loss:.5f} | Val Loss = {val_loss:.5f} | Time = {t1-t0:.1f}s")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_dir / 'st_gnn_checkpoint.pt')
            
    print(f"\n[OK] Training completed. Best validation loss: {best_val_loss:.5f}")
    print("  Saved model parameters to st_gnn_checkpoint.pt")
    print("=" * 80)

if __name__ == "__main__":
    main()
