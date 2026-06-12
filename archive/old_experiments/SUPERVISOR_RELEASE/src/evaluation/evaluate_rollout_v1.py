import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import numpy as np
import scipy.io as sio
import json
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from src.models.stgnn_v1.st_gnn_model import ST_GNN, UHSGraphDataset

def main():
    print("=" * 80)
    print("PHASE 6: AUTOREGRESSIVE ROLLOUT EVALUATION")
    print("=" * 80)
    
    project_root = Path(__file__).resolve().parents[2]
    data_raw_dir = project_root / "data" / "raw"
    data_processed_dir = project_root / "data" / "processed"
    checkpoint_dir = project_root / "checkpoints" / "stgnn_v1"
    results_figures_dir = project_root / "results" / "figures"
    results_metrics_dir = project_root / "results" / "metrics"
    
    results_figures_dir.mkdir(parents=True, exist_ok=True)
    results_metrics_dir.mkdir(parents=True, exist_ok=True)
    
    graph_path = data_processed_dir / "mesh_graph.npz"
    stats_path = data_processed_dir / "st_gnn_stats.json"
    checkpoint_path = checkpoint_dir / "st_gnn_checkpoint.pt"
    plot_dir = results_figures_dir
    plot_dir.mkdir(exist_ok=True)
    
    if not (graph_path.exists() and stats_path.exists() and checkpoint_path.exists()):
        raise FileNotFoundError("Prerequisite files (graph, stats, or GNN checkpoint) missing!")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Scaler Stats and Graph Data
    with open(stats_path, 'r') as f:
        stats = json.load(f)
        
    scaler_mean = np.array(stats['feature_mean'], dtype=np.float32)
    scaler_std = np.array(stats['feature_std'], dtype=np.float32)
    target_dp_std = stats['target_dp_std']
    target_dsg_std = stats['target_dsg_std']
    
    graph_data = np.load(graph_path)
    edge_index = torch.from_numpy(graph_data['edge_index']).to(device)
    edge_trans = torch.from_numpy(graph_data['edge_trans']).to(device)
    dist_to_inj = graph_data['dist_to_inj']
    dist_to_prod = graph_data['dist_to_prod']
    inj_cells = graph_data['inj_cells']
    prod_cells = graph_data['prod_cells']
    
    X_coords = graph_data['X_coords']
    Z_coords = graph_data['Z_coords']
    n_cells = len(X_coords)
    
    # 2. Instantiate and Load Model
    in_features = scaler_mean.shape[0]
    model = ST_GNN(in_features=in_features, hidden_features=64, out_features=2)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    print("[OK] ST-GNN Model loaded successfully.")
    
    # 3. Load Case 5 for Ground Truth Rollout
    print("\nLoading Case 5 ground truth data...")
    case_path = data_raw_dir / "Case_0005_wj.mat"
    case_data = sio.loadmat(str(case_path))
    
    P_gt = case_data['P_matrix'].astype(np.float32)
    Sg_gt = case_data['Sg_matrix'].astype(np.float32)
    Sw_gt = case_data['Sw_matrix'].astype(np.float32)
    xH2_gt = case_data['xH2_matrix'].astype(np.float32)
    yH2_gt = case_data['yH2_matrix'].astype(np.float32)
    perm = case_data['Perm_initial'].squeeze().astype(np.float32)
    poro = case_data['Poro_initial'].squeeze().astype(np.float32)
    
    inj_BHP = case_data['inj_BHP'].squeeze().astype(np.float32)
    prod_BHP = case_data['prod_BHP'].squeeze().astype(np.float32)
    inj_H2 = case_data['inj_H2'].squeeze().astype(np.float32)
    prod_H2 = case_data['prod_H2'].squeeze().astype(np.float32)
    inj_status = case_data['inj_status'].squeeze().astype(np.float32)
    prod_status = case_data['prod_status'].squeeze().astype(np.float32)
    
    n_cells, n_steps = P_gt.shape
    
    # Initialize rollout arrays
    P_pred = np.copy(P_gt)
    Sg_pred = np.copy(Sg_gt)
    
    # Compute elapsed time counters
    time_since_inj = np.zeros(n_steps)
    time_since_prod = np.zeros(n_steps)
    last_inj_t = 0
    last_prod_t = 0
    for t in range(n_steps):
        if inj_status[t] == 1:
            last_inj_t = t
        if prod_status[t] == 1:
            last_prod_t = t
        time_since_inj[t] = t - last_inj_t
        time_since_prod[t] = t - last_prod_t
        
    print(f"\nStarting 146-step Autoregressive Rollout Forecasting (Seeding t=0,1,2)...")
    
    # History queues for autoregressive loops
    dp_history = [
        P_gt[:, 1] - P_gt[:, 0], # t=1
        P_gt[:, 2] - P_gt[:, 1], # t=2
        P_gt[:, 3] - P_gt[:, 2]  # t=3 (will be updated)
    ]
    
    rollout_rmse_P = []
    rollout_rmse_Sg = []
    
    # Run loop
    for t in range(3, n_steps - 1):
        # 1. Prepare current state inputs from predictions
        p_t = P_pred[:, t]
        sg_t = Sg_pred[:, t]
        
        # Consistent two-phase saturation mapping
        sw_t = np.clip(1.0 - sg_t, 0.0, 1.0)
        
        # Compositions can remain close to current ground truth or be approximated
        xh2_t = xH2_gt[:, t]
        yh2_t = yH2_gt[:, t]
        
        # 2. Get history from prediction logs
        dp_t0 = dp_history[-1]
        dp_t1 = dp_history[-2]
        dp_t2 = dp_history[-3]
        
        # 3. Dynamic well controls mapped cell-wise
        inj_BHP_node = np.zeros(n_cells)
        inj_H2_node = np.zeros(n_cells)
        inj_status_node = np.zeros(n_cells)
        
        prod_BHP_node = np.zeros(n_cells)
        prod_H2_node = np.zeros(n_cells)
        prod_status_node = np.zeros(n_cells)
        
        inj_BHP_node[inj_cells] = inj_BHP[t]
        inj_H2_node[inj_cells] = inj_H2[t]
        inj_status_node[inj_cells] = inj_status[t]
        
        prod_BHP_node[prod_cells] = prod_BHP[t]
        prod_H2_node[prod_cells] = prod_H2[t]
        prod_status_node[prod_cells] = prod_status[t]
        
        # Stack feature array
        feat = np.column_stack((
            perm, poro, dist_to_inj, dist_to_prod,
            p_t, sg_t, sw_t, xh2_t, yh2_t,
            dp_t0, dp_t1, dp_t2,
            time_since_inj[t] * np.ones(n_cells),
            time_since_prod[t] * np.ones(n_cells),
            inj_BHP_node, inj_H2_node, inj_status_node,
            prod_BHP_node, prod_H2_node, prod_status_node
        ))
        
        # Standardize features
        feat_scaled = (feat - scaler_mean) / scaler_std
        x_tensor = torch.from_numpy(feat_scaled.astype(np.float32)).to(device)
        
        # Predict
        with torch.no_grad():
            pred_norm = model(x_tensor, edge_index, edge_trans)
            pred_norm = pred_norm.cpu().numpy()
            
        # Denormalize predictions
        dp_pred = pred_norm[:, 0] * target_dp_std
        dsg_pred = pred_norm[:, 1] * target_dsg_std
        
        # Update rollout states
        P_pred[:, t+1] = P_pred[:, t] + dp_pred
        Sg_pred[:, t+1] = np.clip(Sg_pred[:, t] + dsg_pred, 0.0, 1.0)
        
        # Update history queue
        dp_history.append(dp_pred)
        dp_history.pop(0) # Keep queue size 3
        
        # Evaluate errors at next step
        rmse_p = np.sqrt(np.mean((P_pred[:, t+1] - P_gt[:, t+1])**2))
        rmse_sg = np.sqrt(np.mean((Sg_pred[:, t+1] - Sg_gt[:, t+1])**2))
        
        rollout_rmse_P.append(float(rmse_p))
        rollout_rmse_Sg.append(float(rmse_sg))
        
        if (t+1) % 20 == 0 or t == n_steps - 2:
            print(f"  Step {t+1:3d}/{n_steps} | Pressure RMSE = {rmse_p:5.2f} bar | Saturation RMSE = {rmse_sg:6.4f}")
            
    # 4. Generate Performance Plots
    print("\nGenerating performance plots...")
    steps = np.arange(n_steps)
    
    # Track Average, Max, and Min pressure
    avg_P_gt = np.mean(P_gt, axis=0)
    avg_P_pred = np.mean(P_pred, axis=0)
    max_P_gt = np.max(P_gt, axis=0)
    max_P_pred = np.max(P_pred, axis=0)
    min_P_gt = np.min(P_gt, axis=0)
    min_P_pred = np.min(P_pred, axis=0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, avg_P_gt, 'k-', label="Sim Average")
    plt.plot(steps, avg_P_pred, 'k--', label="GNN Average")
    plt.plot(steps, max_P_gt, 'r-', label="Sim Max")
    plt.plot(steps, max_P_pred, 'r--', label="GNN Max")
    plt.plot(steps, min_P_gt, 'b-', label="Sim Min")
    plt.plot(steps, min_P_pred, 'b--', label="GNN Min")
    
    # Operational phase shaded regions
    plt.axvspan(0, 30, color='green', alpha=0.1, label='Inj Phase 1')
    plt.axvspan(30, 43, color='gray', alpha=0.1, label='Shut-in 1')
    plt.axvspan(43, 73, color='orange', alpha=0.1, label='Prod Phase 1')
    plt.axvspan(73, 116, color='green', alpha=0.15, label='Inj Phase 2/Shut-in')
    plt.axvspan(116, 145, color='orange', alpha=0.15, label='Prod Phase 2')
    
    plt.xlabel("Timestep (5-day intervals)")
    plt.ylabel("Pressure (bar)")
    plt.title("Pressure Tracking: ST-GNN Autoregressive Rollout vs Simulator")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(plot_dir / "pressure_rollout_tracking.png", dpi=150)
    plt.close()
    print(f"  Saved pressure tracking to {plot_dir / 'pressure_rollout_tracking.png'}")
    
    # Save overall summary metrics
    summary = {
        'case': 'Case_0005_wj.mat',
        'rollout_steps': len(rollout_rmse_P),
        'final_pressure_rmse': float(rollout_rmse_P[-1]),
        'final_saturation_rmse': float(rollout_rmse_Sg[-1]),
        'mean_pressure_rmse_over_rollout': float(np.mean(rollout_rmse_P)),
        'mean_saturation_rmse_over_rollout': float(np.mean(rollout_rmse_Sg)),
        'max_pressure_error': float(np.max(np.abs(P_pred - P_gt))),
        'max_saturation_error': float(np.max(np.abs(Sg_pred - Sg_gt)))
    }
    
    summary_path = data_dir / "st_gnn_rollout_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved rollout statistics summary to {summary_path.name}")
    print("=" * 80)

if __name__ == "__main__":
    main()
