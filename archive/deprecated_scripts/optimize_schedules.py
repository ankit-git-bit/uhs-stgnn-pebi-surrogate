import torch
import numpy as np
import scipy.io as sio
import json
from pathlib import Path
import matplotlib.pyplot as plt

from st_gnn_model import ST_GNN

def run_rollout_with_controls(model, device, X_init, Sg_init, edge_index, edge_trans, 
                              inj_cells, prod_cells, dist_to_inj, dist_to_prod,
                              perm, poro, scaler_mean, scaler_std, target_dp_std, target_dsg_std,
                              inj_BHP_schedule, prod_BHP_schedule, 
                              inj_status_schedule, prod_status_schedule,
                              time_since_inj, time_since_prod):
    """
    Runs GNN rollout under custom BHP and status schedules.
    Returns predicted pressure and saturation time series.
    """
    n_cells = len(perm)
    n_steps = len(inj_status_schedule)
    
    P_pred = np.zeros((n_cells, n_steps), dtype=np.float32)
    Sg_pred = np.zeros((n_cells, n_steps), dtype=np.float32)
    
    # Initialize seeds
    P_pred[:, :3] = X_init[:, :3]
    Sg_pred[:, :3] = Sg_init[:, :3]
    
    dp_history = [
        P_pred[:, 1] - P_pred[:, 0],
        P_pred[:, 2] - P_pred[:, 1],
        P_pred[:, 3] - P_pred[:, 2] # Dummy placeholder, will be updated
    ]
    
    for t in range(3, n_steps - 1):
        p_t = P_pred[:, t]
        sg_t = Sg_pred[:, t]
        sw_t = np.clip(1.0 - sg_t, 0.0, 1.0)
        
        # Approximate composition fields based on saturation
        xh2_t = np.clip(sg_t * 0.05, 0.0, 0.05)
        yh2_t = np.clip(sg_t * 0.95, 0.0, 0.95)
        
        dp_t0 = dp_history[-1]
        dp_t1 = dp_history[-2]
        dp_t2 = dp_history[-3]
        
        # Apply well BHP schedules mapped node-wise
        inj_BHP_node = np.zeros(n_cells)
        inj_H2_node = np.zeros(n_cells) # Rate is dynamic, starts at 0
        inj_status_node = np.zeros(n_cells)
        
        prod_BHP_node = np.zeros(n_cells)
        prod_H2_node = np.zeros(n_cells)
        prod_status_node = np.zeros(n_cells)
        
        inj_BHP_node[inj_cells] = inj_BHP_schedule[t]
        inj_status_node[inj_cells] = inj_status_schedule[t]
        
        prod_BHP_node[prod_cells] = prod_BHP_schedule[t]
        prod_status_node[prod_cells] = prod_status_schedule[t]
        
        # Feature vector construction
        feat = np.column_stack((
            perm, poro, dist_to_inj, dist_to_prod,
            p_t, sg_t, sw_t, xh2_t, yh2_t,
            dp_t0, dp_t1, dp_t2,
            time_since_inj[t] * np.ones(n_cells),
            time_since_prod[t] * np.ones(n_cells),
            inj_BHP_node, inj_H2_node, inj_status_node,
            prod_BHP_node, prod_H2_node, prod_status_node
        ))
        
        feat_scaled = (feat - scaler_mean) / scaler_std
        x_tensor = torch.from_numpy(feat_scaled.astype(np.float32)).to(device)
        
        with torch.no_grad():
            pred_norm = model(x_tensor, edge_index, edge_trans)
            pred_norm = pred_norm.cpu().numpy()
            
        dp_pred = pred_norm[:, 0] * target_dp_std
        dsg_pred = pred_norm[:, 1] * target_dsg_std
        
        # Rollout update
        P_pred[:, t+1] = P_pred[:, t] + dp_pred
        Sg_pred[:, t+1] = np.clip(Sg_pred[:, t] + dsg_pred, 0.0, 1.0)
        
        dp_history.append(dp_pred)
        dp_history.pop(0)
        
    return P_pred, Sg_pred

def main():
    print("=" * 80)
    print("PHASES 8 & 9: MULTI-VARIABLE INVENTORY & SCHEDULING OPTIMIZATION")
    print("=" * 80)
    
    data_dir = Path(__file__).parent
    graph_path = data_dir / "mesh_graph.npz"
    stats_path = data_dir / "st_gnn_stats.json"
    checkpoint_path = data_dir / "st_gnn_checkpoint.pt"
    
    if not (graph_path.exists() and stats_path.exists() and checkpoint_path.exists()):
        raise FileNotFoundError("Prerequisite training or graph files missing!")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load scaler details and GNN model
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
    cell_area = graph_data['cell_area']
    
    in_features = scaler_mean.shape[0]
    model = ST_GNN(in_features=in_features, hidden_features=64, out_features=2)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Load Case 5 for base configurations
    case_path = data_dir / "Case_0005_wj.mat"
    case_data = sio.loadmat(str(case_path))
    P_init = case_data['P_matrix'].astype(np.float32)
    Sg_init = case_data['Sg_matrix'].astype(np.float32)
    perm = case_data['Perm_initial'].squeeze().astype(np.float32)
    poro = case_data['Poro_initial'].squeeze().astype(np.float32)
    
    # Get baseline schedules
    inj_status = case_data['inj_status'].squeeze().astype(np.float32)
    prod_status = case_data['prod_status'].squeeze().astype(np.float32)
    n_steps = len(inj_status)
    
    # Precompute elapsed time schedules
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
        
    # Productivity indices for well flow rate calculations
    inj_PI = 8.5   # m3 / (day * bar)
    prod_PI = 12.0 # m3 / (day * bar)
    
    # Define BHP candidates to evaluate for optimization
    inj_BHP_levels = [160.0, 180.0, 200.0]  # Injection pressure levels
    prod_BHP_levels = [80.0, 100.0, 120.0]  # Production pressure levels
    
    print("\nRunning GNN surrogate rollout sweep over operational BHP controls...")
    
    results = []
    
    for ibhp in inj_BHP_levels:
        for pbhp in prod_BHP_levels:
            # Construct active schedules
            inj_BHP_schedule = np.zeros(n_steps)
            prod_BHP_schedule = np.zeros(n_steps)
            
            # Injection phases
            inj_BHP_schedule[inj_status == 1] = ibhp
            # Production phases
            prod_BHP_schedule[prod_status == 1] = pbhp
            
            P_pred, Sg_pred = run_rollout_with_controls(
                model, device, P_init, Sg_init, edge_index, edge_trans,
                inj_cells, prod_cells, dist_to_inj, dist_to_prod,
                perm, poro, scaler_mean, scaler_std, target_dp_std, target_dsg_std,
                inj_BHP_schedule, prod_BHP_schedule,
                inj_status, prod_status, time_since_inj, time_since_prod
            )
            
            # 2. Calculate resulting flow rates based on pressure drawdowns
            # P_well is the average pressure of well cells
            inj_P_well = np.mean(P_pred[inj_cells, :], axis=0)
            prod_P_well = np.mean(P_pred[prod_cells, :], axis=0)
            
            q_inj = np.zeros(n_steps)
            q_prod = np.zeros(n_steps)
            
            # Darcy well flow equation: q = PI * (BHP - P_well) for injection, (P_well - BHP) for production
            q_inj[inj_status == 1] = inj_PI * (ibhp - inj_P_well[inj_status == 1])
            q_prod[prod_status == 1] = prod_PI * (prod_P_well[prod_status == 1] - pbhp)
            
            # Clip negative flow rates (prevent reverse flow)
            q_inj = np.clip(q_inj, 0, None)
            q_prod = np.clip(q_prod, 0, None)
            
            # Calculate inventory over time (proxy: sum of Area * Poro * Sg * P)
            # Normalizing by total cells to get an index
            inv_t = []
            for t in range(n_steps):
                inv = np.sum(cell_area * poro * Sg_pred[:, t] * P_pred[:, t])
                inv_t.append(float(inv))
                
            # Recovery efficiency calculation
            total_inj = float(np.sum(q_inj) * 5)  # 5-day intervals
            total_prod = float(np.sum(q_prod) * 5)
            efficiency = total_prod / (total_inj + 1e-10)
            
            print(f"  Inj BHP = {ibhp:.0f} | Prod BHP = {pbhp:.0f} | Total Inj = {total_inj:8.1f} | Total Prod = {total_prod:8.1f} | Efficiency = {efficiency*100:5.2f}%")
            
            results.append({
                'inj_bhp': ibhp,
                'prod_bhp': pbhp,
                'total_injected': total_inj,
                'total_produced': total_prod,
                'efficiency': efficiency,
                'final_inventory': inv_t[-1],
                'inventory_history': inv_t
            })
            
    # Find optimal schedule
    best_run = max(results, key=lambda x: x['efficiency'])
    print(f"\n[OK] Optimization complete!")
    print(f"  Optimal Schedule: Injection BHP = {best_run['inj_bhp']:.1f} bar | Production BHP = {best_run['prod_bhp']:.1f} bar")
    print(f"  Maximum H2 Recovery Efficiency: {best_run['efficiency']*100:.2f}%")
    print(f"  Resulting Gas Saturation Inventory Index: {best_run['final_inventory']:.2f}")
    
    # Save optimization report
    opt_summary_path = data_dir / "st_gnn_optimization_summary.json"
    with open(opt_summary_path, 'w') as f:
        json.dump({
            'optimal_config': {
                'inj_bhp': best_run['inj_bhp'],
                'prod_bhp': best_run['prod_bhp'],
                'max_efficiency': best_run['efficiency']
            },
            'all_sweeps': [{k: v for k, v in r.items() if k != 'inventory_history'} for r in results]
        }, f, indent=2)
    print(f"  Saved optimization configurations to {opt_summary_path.name}")
    
    # Plot inventory evolution for optimal run
    plt.figure(figsize=(8, 5))
    plt.plot(range(n_steps), best_run['inventory_history'], 'b-', label='H2 Gas Inventory Proxy')
    plt.xlabel('Timestep (5-day intervals)')
    plt.ylabel('H2 Inventory Index (bar * m^2)')
    plt.title('Hydrogen Inventory Evolution (Optimal Control Schedule)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(data_dir / 'phase6_plots' / 'h2_inventory_tracking.png', dpi=150)
    plt.close()
    print(f"  Saved inventory tracking plot to phase6_plots/h2_inventory_tracking.png")
    print("=" * 80)

if __name__ == "__main__":
    main()
