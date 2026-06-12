import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

"""
Evaluate ST-GNN v2: Side-by-side Autoregressive Rollout Comparison
====================================================================
Runs 142-step autoregressive rollout for BOTH v1 (st_gnn_checkpoint.pt)
and v2 (st_gnn_v2_checkpoint.pt) on Case 5 (held-out test).

Generates:
  - phase6_plots/v1_vs_v2_pressure_rmse.png   : RMSE over rollout timesteps
  - phase6_plots/v1_vs_v2_pressure_tracking.png: avg/min/max pressure tracking
  - st_gnn_v2_rollout_summary.json             : numerical results
"""

import torch
import numpy as np
import scipy.io as sio
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from st_gnn_model   import ST_GNN
from src.models.stgnn_v2.st_gnn_v2_model import ST_GNN_v2


# --- Rollout Engine --------------------------------------------------------

def run_rollout(model, device, case_data, graph_data, scaler_mean, scaler_std,
                dp_std, dsg_std, label="model"):
    """
    142-step autoregressive rollout on Case 5.
    Seeds from ground-truth t=0,1,2 then iterates with model's own predictions.

    Returns:
        P_pred  : (n_cells, n_steps) float32
        Sg_pred : (n_cells, n_steps) float32
        rmse_P  : list[float] -- per-step RMSE
        rmse_Sg : list[float] -- per-step RMSE
    """
    P_gt  = case_data['P_matrix'].astype(np.float32)
    Sg_gt = case_data['Sg_matrix'].astype(np.float32)
    Sw_gt = case_data['Sw_matrix'].astype(np.float32)
    xH2_gt = case_data['xH2_matrix'].astype(np.float32)
    yH2_gt = case_data['yH2_matrix'].astype(np.float32)
    perm  = case_data['Perm_initial'].squeeze().astype(np.float32)
    poro  = case_data['Poro_initial'].squeeze().astype(np.float32)

    inj_BHP    = case_data['inj_BHP'].squeeze().astype(np.float32)
    prod_BHP   = case_data['prod_BHP'].squeeze().astype(np.float32)
    inj_H2     = case_data['inj_H2'].squeeze().astype(np.float32)
    prod_H2    = case_data['prod_H2'].squeeze().astype(np.float32)
    inj_status  = case_data['inj_status'].squeeze().astype(np.float32)
    prod_status = case_data['prod_status'].squeeze().astype(np.float32)

    n_cells, n_steps = P_gt.shape

    edge_index  = torch.from_numpy(graph_data['edge_index']).to(device)
    edge_trans  = torch.from_numpy(graph_data['edge_trans']).to(device)
    dist_to_inj = graph_data['dist_to_inj']
    dist_to_prod = graph_data['dist_to_prod']
    inj_cells   = graph_data['inj_cells']
    prod_cells  = graph_data['prod_cells']

    # -- Elapsed-time counters ---------------------------------------------
    time_since_inj  = np.zeros(n_steps)
    time_since_prod = np.zeros(n_steps)
    last_inj_t = last_prod_t = 0
    for t in range(n_steps):
        if inj_status[t]  == 1: last_inj_t  = t
        if prod_status[t] == 1: last_prod_t = t
        time_since_inj[t]  = t - last_inj_t
        time_since_prod[t] = t - last_prod_t

    # -- Initialize rollout arrays -----------------------------------------
    P_pred  = np.copy(P_gt)
    Sg_pred = np.copy(Sg_gt)

    dp_history = [
        P_gt[:, 1] - P_gt[:, 0],
        P_gt[:, 2] - P_gt[:, 1],
        P_gt[:, 3] - P_gt[:, 2],
    ]

    rmse_P  = []
    rmse_Sg = []

    model.eval()
    print(f"  Running rollout [{label}]...")

    for t in range(3, n_steps - 1):
        p_t  = P_pred[:, t]
        sg_t = Sg_pred[:, t]
        sw_t = np.clip(1.0 - sg_t, 0.0, 1.0)
        xh2_t = xH2_gt[:, t]   # compositions approximated from ground truth
        yh2_t = yH2_gt[:, t]

        dp_t0 = dp_history[-1]
        dp_t1 = dp_history[-2]
        dp_t2 = dp_history[-3]

        # -- Well controls mapped node-wise --------------------------------
        inj_BHP_node     = np.zeros(n_cells)
        inj_H2_node      = np.zeros(n_cells)
        inj_status_node  = np.zeros(n_cells)
        prod_BHP_node    = np.zeros(n_cells)
        prod_H2_node     = np.zeros(n_cells)
        prod_status_node = np.zeros(n_cells)

        inj_BHP_node[inj_cells]    = inj_BHP[t]
        inj_H2_node[inj_cells]     = inj_H2[t]
        inj_status_node[inj_cells] = inj_status[t]
        prod_BHP_node[prod_cells]   = prod_BHP[t]
        prod_H2_node[prod_cells]    = prod_H2[t]
        prod_status_node[prod_cells] = prod_status[t]

        feat = np.column_stack((
            perm, poro, dist_to_inj, dist_to_prod,
            p_t, sg_t, sw_t, xh2_t, yh2_t,
            dp_t0, dp_t1, dp_t2,
            time_since_inj[t]  * np.ones(n_cells),
            time_since_prod[t] * np.ones(n_cells),
            inj_BHP_node, inj_H2_node, inj_status_node,
            prod_BHP_node, prod_H2_node, prod_status_node,
        ))

        feat_scaled = (feat - scaler_mean) / scaler_std
        x_t = torch.from_numpy(feat_scaled.astype(np.float32)).to(device)

        with torch.no_grad():
            pred_norm = model(x_t, edge_index, edge_trans).cpu().numpy()

        dp_pred  = pred_norm[:, 0] * dp_std
        dsg_pred = pred_norm[:, 1] * dsg_std

        P_pred[:, t+1]  = P_pred[:, t]  + dp_pred
        Sg_pred[:, t+1] = np.clip(Sg_pred[:, t] + dsg_pred, 0.0, 1.0)

        dp_history.append(dp_pred)
        dp_history.pop(0)

        err_P  = float(np.sqrt(np.mean((P_pred[:, t+1]  - P_gt[:, t+1])**2)))
        err_Sg = float(np.sqrt(np.mean((Sg_pred[:, t+1] - Sg_gt[:, t+1])**2)))
        rmse_P.append(err_P)
        rmse_Sg.append(err_Sg)

        if (t + 1) % 30 == 0 or t == n_steps - 2:
            print(f"    Step {t+1:3d}/{n_steps} | P-RMSE={err_P:6.2f} bar | Sg-RMSE={err_Sg:.4f}")

    return P_pred, Sg_pred, rmse_P, rmse_Sg


# --- Main ------------------------------------------------------------------

def main():
    print("=" * 80)
    print("PHASE 6v2: AUTOREGRESSIVE ROLLOUT -- ST-GNN v1 vs v2")
    print("=" * 80)

    data_dir   = Path(__file__).parent
    graph_path = data_dir / "mesh_graph.npz"
    plot_dir   = data_dir / "phase6_plots"
    plot_dir.mkdir(exist_ok=True)

    # Required files
    for f in [graph_path,
              data_dir / "st_gnn_stats.json",
              data_dir / "st_gnn_checkpoint.pt",
              data_dir / "st_gnn_v2_stats.json",
              data_dir / "st_gnn_v2_checkpoint.pt"]:
        if not f.exists():
            raise FileNotFoundError(f"Missing required file: {f.name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -- Load shared data --------------------------------------------------
    graph_data = np.load(graph_path)
    case_data  = sio.loadmat(str(data_dir / "Case_0005_wj.mat"))
    n_cells, n_steps = case_data['P_matrix'].shape

    # -- Load v1 model -----------------------------------------------------
    with open(data_dir / "st_gnn_stats.json") as f:
        stats_v1 = json.load(f)
    mean_v1 = np.array(stats_v1['feature_mean'], dtype=np.float32)
    std_v1  = np.array(stats_v1['feature_std'],  dtype=np.float32)
    dp_std_v1  = stats_v1['target_dp_std']
    dsg_std_v1 = stats_v1['target_dsg_std']

    model_v1 = ST_GNN(in_features=mean_v1.shape[0], hidden_features=64, out_features=2)
    model_v1.load_state_dict(torch.load(data_dir / "st_gnn_checkpoint.pt", map_location=device))
    model_v1 = model_v1.to(device)
    print("\n[OK] ST-GNN v1 loaded.")

    # -- Load v2 model -----------------------------------------------------
    with open(data_dir / "st_gnn_v2_stats.json") as f:
        stats_v2 = json.load(f)
    mean_v2 = np.array(stats_v2['feature_mean'], dtype=np.float32)
    std_v2  = np.array(stats_v2['feature_std'],  dtype=np.float32)
    dp_std_v2  = stats_v2['target_dp_std']
    dsg_std_v2 = stats_v2['target_dsg_std']

    model_v2 = ST_GNN_v2(in_features=mean_v2.shape[0], hidden_features=128, out_features=2, dropout=0.0)
    model_v2.load_state_dict(torch.load(data_dir / "st_gnn_v2_checkpoint.pt", map_location=device))
    model_v2 = model_v2.to(device)
    print("[OK] ST-GNN v2 loaded.\n")

    # -- Run rollouts ------------------------------------------------------
    P_gt = case_data['P_matrix'].astype(np.float32)
    Sg_gt = case_data['Sg_matrix'].astype(np.float32)

    P_v1, Sg_v1, rmse_P_v1, rmse_Sg_v1 = run_rollout(
        model_v1, device, case_data, graph_data, mean_v1, std_v1, dp_std_v1, dsg_std_v1, label="v1")
    P_v2, Sg_v2, rmse_P_v2, rmse_Sg_v2 = run_rollout(
        model_v2, device, case_data, graph_data, mean_v2, std_v2, dp_std_v2, dsg_std_v2, label="v2")

    steps = np.arange(4, n_steps)   # Steps 4..146 (rollout starts at t=3)

    # -- Summary statistics ------------------------------------------------
    summary_v2 = {
        'model':                    'ST-GNN v2',
        'case':                     'Case_0005_wj.mat',
        'rollout_steps':            len(rmse_P_v2),
        'mean_pressure_rmse':       float(np.mean(rmse_P_v2)),
        'final_pressure_rmse':      float(rmse_P_v2[-1]),
        'mean_saturation_rmse':     float(np.mean(rmse_Sg_v2)),
        'final_saturation_rmse':    float(rmse_Sg_v2[-1]),
        'max_pressure_error':       float(np.max(np.abs(P_v2 - P_gt))),
        'max_saturation_error':     float(np.max(np.abs(Sg_v2 - Sg_gt))),
    }
    comparison = {
        'v1': {
            'mean_pressure_rmse':  float(np.mean(rmse_P_v1)),
            'final_pressure_rmse': float(rmse_P_v1[-1]),
            'mean_saturation_rmse': float(np.mean(rmse_Sg_v1)),
        },
        'v2': {
            'mean_pressure_rmse':  float(np.mean(rmse_P_v2)),
            'final_pressure_rmse': float(rmse_P_v2[-1]),
            'mean_saturation_rmse': float(np.mean(rmse_Sg_v2)),
        },
        'improvement_percent': {
            'mean_pressure_rmse': float(
                100.0 * (np.mean(rmse_P_v1) - np.mean(rmse_P_v2)) / np.mean(rmse_P_v1)),
            'final_pressure_rmse': float(
                100.0 * (rmse_P_v1[-1] - rmse_P_v2[-1]) / rmse_P_v1[-1]),
        }
    }

    with open(results_metrics_dir / "st_gnn_v2_rollout_summary.json", 'w') as f:
        json.dump({**summary_v2, 'comparison': comparison}, f, indent=2)
    print(f"\nSaved -> st_gnn_v2_rollout_summary.json")

    # -- Print comparison --------------------------------------------------
    print("\n" + "=" * 60)
    print("ROLLOUT COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<35} {'v1':>10} {'v2':>10} {'Delta%':>8}")
    print("-" * 60)
    for metric, key_v1, key_v2 in [
        ("Mean Pressure RMSE (bar)",    np.mean(rmse_P_v1),   np.mean(rmse_P_v2)),
        ("Final Pressure RMSE (bar)",   rmse_P_v1[-1],        rmse_P_v2[-1]),
        ("Mean Saturation RMSE",        np.mean(rmse_Sg_v1),  np.mean(rmse_Sg_v2)),
        ("Final Saturation RMSE",       rmse_Sg_v1[-1],       rmse_Sg_v2[-1]),
    ]:
        delta = 100.0 * (key_v1 - key_v2) / (key_v1 + 1e-12)
        sign  = "?" if delta > 0 else "?"
        print(f"  {metric:<33} {key_v1:>10.4f} {key_v2:>10.4f} {sign}{abs(delta):>6.1f}%")
    print("=" * 60)

    # -- Plot 1: RMSE over rollout -----------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("ST-GNN v1 vs v2: Autoregressive Rollout Performance\n(Case 5, 142 steps)",
                 fontsize=14, fontweight='bold')

    ax = axes[0]
    ax.plot(steps, rmse_P_v1, 'tomato',    lw=1.8, label=f"v1  (mean={np.mean(rmse_P_v1):.1f} bar)")
    ax.plot(steps, rmse_P_v2, 'steelblue', lw=1.8, label=f"v2  (mean={np.mean(rmse_P_v2):.1f} bar)")
    ax.axhline(np.mean(rmse_P_v1), color='tomato',    lw=1, ls='--', alpha=0.5)
    ax.axhline(np.mean(rmse_P_v2), color='steelblue', lw=1, ls='--', alpha=0.5)
    # Operational phase shading
    for span, col, lbl in [((3, 30),  'green',  'Inj 1'),
                            ((30, 43), 'gray',   'Shut-in 1'),
                            ((43, 73), 'orange', 'Prod 1'),
                            ((73,116), 'green',  'Inj 2'),
                            ((116,145),'orange', 'Prod 2')]:
        ax.axvspan(*span, color=col, alpha=0.07, label=lbl)
    ax.set_ylabel("Pressure RMSE (bar)", fontsize=11)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(steps, rmse_Sg_v1, 'tomato',    lw=1.8, label=f"v1  (mean={np.mean(rmse_Sg_v1):.4f})")
    ax.plot(steps, rmse_Sg_v2, 'steelblue', lw=1.8, label=f"v2  (mean={np.mean(rmse_Sg_v2):.4f})")
    ax.axhline(np.mean(rmse_Sg_v1), color='tomato',    lw=1, ls='--', alpha=0.5)
    ax.axhline(np.mean(rmse_Sg_v2), color='steelblue', lw=1, ls='--', alpha=0.5)
    ax.set_xlabel("Timestep (5-day intervals)", fontsize=11)
    ax.set_ylabel("Saturation RMSE", fontsize=11)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(plot_dir / "v1_vs_v2_rollout_rmse.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved -> phase6_plots/v1_vs_v2_rollout_rmse.png")

    # -- Plot 2: Pressure tracking (avg/min/max) ---------------------------
    t_axis = np.arange(n_steps)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    fig.suptitle("Field-Average Pressure Tracking: ST-GNN v1 vs v2", fontsize=13, fontweight='bold')

    for ax, P_pred, label, color in [
        (axes[0], P_v1, "v1 (12 epochs, hidden=64)", "tomato"),
        (axes[1], P_v2, "v2 (50 epochs, hidden=128, residual)", "steelblue"),
    ]:
        ax.fill_between(t_axis, np.min(P_gt, axis=0), np.max(P_gt, axis=0),
                        color='gray', alpha=0.15, label='GT range')
        ax.plot(t_axis, np.mean(P_gt,   axis=0), 'k-',  lw=2, label='Simulator avg')
        ax.plot(t_axis, np.max(P_gt,    axis=0), 'k--', lw=1, alpha=0.5)
        ax.plot(t_axis, np.min(P_gt,    axis=0), 'k--', lw=1, alpha=0.5)
        ax.plot(t_axis, np.mean(P_pred, axis=0), color=color, lw=2, ls='--', label=f'{label} avg')
        ax.plot(t_axis, np.max(P_pred,  axis=0), color=color, lw=1, ls=':')
        ax.plot(t_axis, np.min(P_pred,  axis=0), color=color, lw=1, ls=':')
        for span, col in [((0, 30),'green'),((30,43),'gray'),((43,73),'orange'),
                          ((73,116),'green'),((116,145),'orange')]:
            ax.axvspan(*span, color=col, alpha=0.06)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Pressure (bar)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_title(label, fontsize=10)

    plt.tight_layout()
    fig.savefig(plot_dir / "v1_vs_v2_pressure_tracking.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved -> phase6_plots/v1_vs_v2_pressure_tracking.png")

    print("\n[DONE] Evaluation complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
