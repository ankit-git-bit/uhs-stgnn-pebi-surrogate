import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import time
import json

from st_gnn_model import UHSGraphDataset, ST_GNN

class PhysicsLossModule(nn.Module):
    """
    Computes physical mass conservation and pressure diffusion residuals:
      Residual_i = Area_i * Poro_i * dP_pred_i - gamma * NetFlow_i - beta * q_i
    """
    def __init__(self, init_gamma=0.01, init_beta=0.1):
        super().__init__()
        # Learnable physical scaling coefficients to align flow, accumulation, and well rate units
        self.gamma = nn.Parameter(torch.tensor(init_gamma, dtype=torch.float32, requires_grad=True))
        self.beta = nn.Parameter(torch.tensor(init_beta, dtype=torch.float32, requires_grad=True))
        
    def forward(self, dp_pred, P_curr, phi, cell_area, edge_index, edge_trans, q_well):
        """
        dp_pred: predicted pressure change (N,)
        P_curr: current pressure (N,)
        phi: porosity (N,)
        cell_area: Voronoi cell area (N,)
        edge_index: graph edges (2, E)
        edge_trans: transmissibilities (E,)
        q_well: well rate (N,)
        """
        row, col = edge_index[0], edge_index[1]
        
        # 1. Accumulation Term: V_i * phi_i * dP_i
        # Thickness is assumed uniform, so volume is proportional to area
        accumulation = cell_area * phi * dp_pred
        
        # 2. Darcy Flow Term: Sum_{j in N(i)} T_ij * (P_j - P_i)
        dp_edge = P_curr[col] - P_curr[row] # P_j - P_i
        flow_edge = edge_trans * dp_edge
        
        net_flow = torch.zeros_like(P_curr)
        net_flow.index_add_(0, row, flow_edge)
        
        # 3. Compute Residual
        # R_i = Accumulation_i - gamma * Flow_i - beta * q_i
        residual = accumulation - self.gamma * net_flow - self.beta * q_well
        
        # Mean squared residual loss
        physics_loss = torch.mean(residual**2)
        return physics_loss

def evaluate_pignn(model, dataset, device, target_dp_std, target_dsg_std):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for i in range(len(dataset)):
            x, target, _ = dataset.get_scaled_features(i)
            x = x.to(device)
            target = target.to(device)
            
            target_norm = torch.zeros_like(target)
            target_norm[:, 0] = target[:, 0] / target_dp_std
            target_norm[:, 1] = target[:, 1] / target_dsg_std
            
            pred_norm = model(x, dataset.edge_index.to(device), dataset.edge_trans.to(device))
            loss = torch.mean((pred_norm - target_norm)**2)
            total_loss += loss.item()
            
    return total_loss / len(dataset)

def main():
    print("=" * 80)
    print("PHASE 7: PHYSICS-INFORMED GRAPH NEURAL NETWORK (PIGNN)")
    print("=" * 80)
    
    data_dir = Path(__file__).parent
    graph_path = data_dir / "mesh_graph.npz"
    stats_path = data_dir / "st_gnn_stats.json"
    
    if not (graph_path.exists() and stats_path.exists()):
        raise FileNotFoundError("Run graph_construction.py and train_st_gnn.py first!")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load scaling parameters
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    target_dp_std = stats['target_dp_std']
    target_dsg_std = stats['target_dsg_std']
    
    # Load graph details
    graph_data = np.load(graph_path)
    cell_area = torch.from_numpy(graph_data['cell_area']).to(device)
    
    # 1. Load Datasets
    print("\nLoading Training Dataset (Cases 1-4)...")
    train_dataset = UHSGraphDataset(data_dir, case_ids=[1, 2, 3, 4], graph_path=graph_path, stride=4)
    print("\nLoading Validation Dataset (Case 5)...")
    val_dataset = UHSGraphDataset(data_dir, case_ids=[5], graph_path=graph_path, scaler=train_dataset.scaler)
    
    # 2. Instantiate Model and Physics Loss Module
    in_features = train_dataset.scaler['mean'].shape[0]
    model = ST_GNN(in_features=in_features, hidden_features=64, out_features=2)
    model = model.to(device)
    
    physics_module = PhysicsLossModule(init_gamma=0.01, init_beta=0.1).to(device)
    
    optimizer = optim.Adam([
        {'params': model.parameters(), 'lr': 1e-3, 'weight_decay': 1e-5},
        {'params': physics_module.parameters(), 'lr': 1e-2} # Faster learning for physical parameters
    ])
    
    # Loss weights
    lambda_physics = 0.05 # Regularization coefficient
    
    epochs = 12
    batch_size = 16
    n_samples = len(train_dataset)
    indices = np.arange(n_samples)
    
    best_val_loss = float('inf')
    
    print(f"\nTraining PIGNN for {epochs} epochs with physics loss weight = {lambda_physics}...")
    
    for epoch in range(1, epochs + 1):
        model.train()
        np.random.shuffle(indices)
        
        t0 = time.time()
        epoch_data_loss = 0.0
        epoch_phys_loss = 0.0
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            optimizer.zero_grad()
            batch_loss = 0.0
            
            for idx in batch_indices:
                x, target, sample_info = train_dataset.get_scaled_features(idx)
                x = x.to(device)
                target = target.to(device)
                
                # Normalize target
                target_norm = torch.zeros_like(target)
                target_norm[:, 0] = target[:, 0] / target_dp_std
                target_norm[:, 1] = target[:, 1] / target_dsg_std
                
                # Forward
                pred_norm = model(x, train_dataset.edge_index.to(device), train_dataset.edge_trans.to(device))
                loss_data = torch.mean((pred_norm - target_norm)**2)
                
                # Compute physical properties for PINN loss
                dp_pred = pred_norm[:, 0] * target_dp_std # Denormalized P change (bar)
                
                # Get current pressure
                P_curr = torch.from_numpy(sample_info['P_t']).to(device)
                
                # Porosity node features
                phi = x[:, 1] * train_dataset.scaler['std'][1] + train_dataset.scaler['mean'][1]
                
                # Well rate node features (inj_H2 is feature 15, prod_H2 is feature 18)
                inj_rate = x[:, 15] * train_dataset.scaler['std'][15] + train_dataset.scaler['mean'][15]
                prod_rate = x[:, 18] * train_dataset.scaler['std'][18] + train_dataset.scaler['mean'][18]
                q_well = inj_rate - prod_rate
                
                # Compute physical residual
                loss_phys = physics_module(
                    dp_pred, P_curr, phi, cell_area,
                    train_dataset.edge_index.to(device),
                    train_dataset.edge_trans.to(device),
                    q_well
                )
                
                # Combined Loss
                loss_total = loss_data + lambda_physics * loss_phys
                loss_total.backward()
                
                batch_loss += loss_total.item()
                epoch_data_loss += loss_data.item()
                epoch_phys_loss += loss_phys.item()
                
            optimizer.step()
            
        epoch_data_loss /= n_samples
        epoch_phys_loss /= n_samples
        
        val_loss = evaluate_pignn(model, val_dataset, device, target_dp_std, target_dsg_std)
        t1 = time.time()
        
        print(f"  Epoch {epoch:2d}/{epochs:2d} | Data Loss = {epoch_data_loss:.5f} | Phys Loss = {epoch_phys_loss:.5f} | Val Loss = {val_loss:.5f} | Time = {t1-t0:.1f}s")
        print(f"    Learned Physical Constants: gamma = {physics_module.gamma.item():.6f} | beta = {physics_module.beta.item():.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), data_dir / 'st_gnn_pignn_checkpoint.pt')
            
    print(f"\n[OK] PIGNN Training completed. Best validation loss: {best_val_loss:.5f}")
    print("  Saved physics-informed parameters to st_gnn_pignn_checkpoint.pt")
    print("=" * 80)

if __name__ == "__main__":
    main()
