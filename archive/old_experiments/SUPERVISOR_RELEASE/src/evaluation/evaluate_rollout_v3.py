import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

"""
evaluate_rollout_v3.py
======================
Comprehensive rollout evaluation: ST-GNN v1 vs v2 vs v3
Reports 1-step, 10-step, 50-step, 146-step RMSE.
Generates 5 publication-quality figures.
"""

import torch
import numpy as np
import scipy.io as sio
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

from src.models.stgnn_v1.st_gnn_model import ST_GNN
from src.models.stgnn_v2.st_gnn_v2_model import ST_GNN_v2


# ---------------------------------------------------------------------------
# Rollout engine
# ---------------------------------------------------------------------------

def run_rollout(model, device, case_data, graph_data,
                scaler_mean, scaler_std, dp_std, dsg_std, label=""):
    """
    142-step autoregressive rollout on Case 5.
    Returns: (P_pred, Sg_pred, rmse_P_list, rmse_Sg_list)
    """
    P_gt    = case_data['P_matrix'].astype(np.float32)
    Sg_gt   = case_data['Sg_matrix'].astype(np.float32)
    xH2_gt  = case_data['xH2_matrix'].astype(np.float32)
    yH2_gt  = case_data['yH2_matrix'].astype(np.float32)
    perm    = case_data['Perm_initial'].squeeze().astype(np.float32)
    poro    = case_data['Poro_initial'].squeeze().astype(np.float32)
    inj_BHP    = case_data['inj_BHP'].squeeze().astype(np.float32)
    prod_BHP   = case_data['prod_BHP'].squeeze().astype(np.float32)
    inj_H2     = case_data['inj_H2'].squeeze().astype(np.float32)
    prod_H2    = case_data['prod_H2'].squeeze().astype(np.float32)
    inj_status  = case_data['inj_status'].squeeze().astype(np.float32)
    prod_status = case_data['prod_status'].squeeze().astype(np.float32)

    n_cells, n_steps = P_gt.shape
    ei = torch.from_numpy(graph_data['edge_index']).to(device)
    et = torch.from_numpy(graph_data['edge_trans']).to(device)
    dist_to_inj  = graph_data['dist_to_inj']
    dist_to_prod = graph_data['dist_to_prod']
    inj_cells    = graph_data['inj_cells']
    prod_cells   = graph_data['prod_cells']

    # Elapsed time counters
    t_inj  = np.zeros(n_steps)
    t_prod = np.zeros(n_steps)
    li = lp = 0
    for t in range(n_steps):
        if inj_status[t]  == 1: li = t
        if prod_status[t] == 1: lp = t
        t_inj[t]  = t - li
        t_prod[t] = t - lp

    P_pred  = np.copy(P_gt)
    Sg_pred = np.copy(Sg_gt)
    dp_hist = [P_gt[:, 1] - P_gt[:, 0],
               P_gt[:, 2] - P_gt[:, 1],
               P_gt[:, 3] - P_gt[:, 2]]
    rmse_P  = []
    rmse_Sg = []

    model.eval()
    if label:
        print(f"  Running rollout [{label}]...")

    with torch.no_grad():
        for t in range(3, n_steps - 1):
            p_t  = P_pred[:, t]
            sg_t = Sg_pred[:, t]
            sw_t = np.clip(1.0 - sg_t, 0.0, 1.0)
            dp0, dp1, dp2 = dp_hist[-1], dp_hist[-2], dp_hist[-3]

            ib = np.zeros(n_cells); ih = np.zeros(n_cells); ist = np.zeros(n_cells)
            pb = np.zeros(n_cells); ph = np.zeros(n_cells); pst = np.zeros(n_cells)
            ib[inj_cells]   = inj_BHP[t]
            ih[inj_cells]   = inj_H2[t]
            ist[inj_cells]  = inj_status[t]
            pb[prod_cells]  = prod_BHP[t]
            ph[prod_cells]  = prod_H2[t]
            pst[prod_cells] = prod_status[t]

            feat = np.column_stack((
                perm, poro, dist_to_inj, dist_to_prod,
                p_t, sg_t, sw_t, xH2_gt[:, t], yH2_gt[:, t],
                dp0, dp1, dp2,
                t_inj[t]  * np.ones(n_cells),
                t_prod[t] * np.ones(n_cells),
                ib, ih, ist, pb, ph, pst,
            ))
            fs = (feat - scaler_mean) / scaler_std
            x  = torch.from_numpy(fs.astype(np.float32)).to(device)
            pn = model(x, ei, et).cpu().numpy()

            dp_pred  = pn[:, 0] * dp_std
            dsg_pred = pn[:, 1] * dsg_std
            P_pred[:, t+1]  = P_pred[:, t]  + dp_pred
            Sg_pred[:, t+1] = np.clip(Sg_pred[:, t] + dsg_pred, 0.0, 1.0)
            dp_hist.append(dp_pred)
            dp_hist.pop(0)

            rmse_P.append(float(np.sqrt(np.mean((P_pred[:, t+1]  - P_gt[:, t+1])**2))))
            rmse_Sg.append(float(np.sqrt(np.mean((Sg_pred[:, t+1] - Sg_gt[:, t+1])**2))))

            if (t + 1) % 40 == 0 or t == n_steps - 2:
                print(f"    t={t+1:3d}/{n_steps} | P-RMSE={rmse_P[-1]:6.2f} bar | "
                      f"Sg-RMSE={rmse_Sg[-1]:.5f}")

    return P_pred, Sg_pred, rmse_P, rmse_Sg


# ---------------------------------------------------------------------------
# Multi-step summary
# ---------------------------------------------------------------------------

def multistep_summary(rmse_P, rmse_Sg, steps=(1, 10, 50, 146)):
    """Extract RMSE at specific rollout steps."""
    n = len(rmse_P)
    out = {}
    for s in steps:
        i = min(s - 1, n - 1)
        out[s] = {'P': rmse_P[i], 'Sg': rmse_Sg[i]}
    return out


# ---------------------------------------------------------------------------
# Figure 1: Training loss curves
# ---------------------------------------------------------------------------

def fig_training_curves(plot_dir, data_dir):
    histories = {}
    for name, path in [('v2', 'st_gnn_v2_training_history.json'),
                       ('v3', 'st_gnn_v3_training_history.json')]:
        p = data_dir / path
        if p.exists():
            with open(p) as f:
                histories[name] = json.load(f)

    if not histories:
        print("  [SKIP] No training histories found for curve plot")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {'v2': 'steelblue', 'v3': 'darkorange'}
    for name, hist in histories.items():
        epochs = [h['epoch'] for h in hist]
        train  = [h['train_loss'] for h in hist]
        val    = [h['val_loss'] for h in hist if h.get('val_loss') == h.get('val_loss')]
        val_ep = [h['epoch']    for h in hist if h.get('val_loss') == h.get('val_loss')]
        c = colors.get(name, 'gray')
        ax.plot(epochs, train, color=c, lw=1.8, label=f"{name} train")
        ax.scatter(val_ep, val, color=c, marker='o', s=40, label=f"{name} val")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Weighted MSE Loss", fontsize=12)
    ax.set_title("ST-GNN Training Loss Curves: v2 vs v3", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    fig.savefig(plot_dir / "fig8_training_curves.png", dpi=600, bbox_inches='tight')
    fig.savefig(plot_dir / "fig8_training_curves.pdf", bbox_inches='tight')
    plt.close(fig)
    print("  Saved -> fig8_training_curves.png")


# ---------------------------------------------------------------------------
# Figure 2: Rollout RMSE vs time (v1 vs v2 vs v3)
# ---------------------------------------------------------------------------

def fig_rollout_rmse(plot_dir, rollout_data):
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    fig.suptitle("Autoregressive Rollout RMSE: v1 vs v2 vs v3\n(Test Case 5, 142 steps)",
                 fontsize=15, fontweight='bold', y=0.96)

    colors = {'v1': 'tomato', 'v2': 'steelblue', 'v3': 'darkorange'}
    ls     = {'v1': '-',      'v2': '--',         'v3': '-'}

    phase_spans = [(3, 30, 'green', 'Inj 1'),
                   (30, 43, 'gray', 'Shut-in 1'),
                   (43, 73, 'orange', 'Prod 1'),
                   (73, 103, 'green', 'Inj 2'),
                   (103, 116, 'gray', 'Shut-in 2'),
                   (116, 145, 'orange', 'Prod 2')]

    transitions = [
        (30, "Inj 1 Stop", "red"),
        (43, "Prod 1 Start", "blue"),
        (73, "Inj 2 Start / Prod 1 Stop", "purple"),
        (103, "Inj 2 Stop", "red"),
        (116, "Prod 2 Start", "blue")
    ]

    for ax_i, metric in enumerate(['rmse_P', 'rmse_Sg']):
        ax = axes[ax_i]

        for name, rd in rollout_data.items():
            data  = rd[metric]
            steps = np.arange(4, 4 + len(data))
            mean_ = np.mean(data)
            ax.plot(steps, data, color=colors.get(name, 'gray'),
                    lw=2.0, ls=ls.get(name, '-'),
                    label=f"ST-GNN {name} (mean={mean_:.2f})" if metric == 'rmse_P'
                          else f"ST-GNN {name} (mean={mean_:.4f})")
            ax.axhline(mean_, color=colors.get(name, 'gray'), lw=0.9,
                       ls=':', alpha=0.5)
            
            # Dynamic annotation for peak error locations in pressure
            if metric == 'rmse_P' and name in ['v2', 'v3']:
                max_idx = np.argmax(data)
                max_x = steps[max_idx]
                max_y = data[max_idx]
                
                if name == 'v3':
                    ax.annotate(f"v3 Peak: {max_y:.1f} bar", xy=(max_x, max_y), 
                                xytext=(max_x + 12, max_y + 4),
                                arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.0),
                                fontsize=9, fontweight='bold', 
                                bbox=dict(boxstyle='round,pad=0.25', fc='#ffe6cc', edgecolor='darkorange', alpha=0.9))
                elif name == 'v2':
                    ax.annotate(f"v2 Peak: {max_y:.1f} bar", xy=(max_x, max_y), 
                                xytext=(max_x + 12, max_y - 12),
                                arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.0),
                                fontsize=9, fontweight='bold', 
                                bbox=dict(boxstyle='round,pad=0.25', fc='#e6f2ff', edgecolor='steelblue', alpha=0.9))

        # Phase shading
        for s, e, col, lbl in phase_spans:
            ax.axvspan(s, e, color=col, alpha=0.04)

        # Vertical transition lines
        for step, label, col in transitions:
            ax.axvline(step, color=col, ls='--', lw=1.0, alpha=0.6)
            if ax_i == 0:
                ax.text(step - 1.5, ax.get_ylim()[0] + 0.93 * (ax.get_ylim()[1] - ax.get_ylim()[0]), 
                        label, rotation=90, fontsize=8, color=col, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.8, edgecolor='none'))

        # Add stabilization text annotation
        if metric == 'rmse_P':
            ax.text(122, 12, "Rollout Error\nStabilization Region", 
                    fontsize=9, fontstyle='italic', fontweight='bold', color='dimgray',
                    bbox=dict(boxstyle='round,pad=0.3', fc='#f2f2f2', edgecolor='darkgray', alpha=0.8))
        else:
            ax.text(122, 0.006, "Saturation Error\nStabilization", 
                    fontsize=9, fontstyle='italic', fontweight='bold', color='dimgray',
                    bbox=dict(boxstyle='round,pad=0.3', fc='#f2f2f2', edgecolor='darkgray', alpha=0.8))

        # Styling
        ax.tick_params(axis='both', which='major', labelsize=11)
        ax.grid(True, alpha=0.25, linestyle=':')
        
        if ax_i == 0:
            ax.set_ylabel("Pressure RMSE (bar)", fontsize=12, fontweight='bold')
            ax.set_ylim(0, 110)
        else:
            ax.set_ylabel("Saturation RMSE", fontsize=12, fontweight='bold')
            ax.set_xlabel("Timestep (5-day intervals)", fontsize=12, fontweight='bold')
            ax.set_ylim(0, 0.05)

        ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(plot_dir / "fig9_rollout_rmse.png", dpi=600, bbox_inches='tight')
    fig.savefig(plot_dir / "fig9_rollout_rmse.pdf", bbox_inches='tight')
    plt.close(fig)
    print("  Saved -> fig9_rollout_rmse.png & .pdf")


# ---------------------------------------------------------------------------
# Figure 3: Pressure tracking (v3 vs simulator)
# ---------------------------------------------------------------------------

def fig_pressure_tracking(plot_dir, P_gt, P_v2, n_steps):
    t = np.arange(n_steps)
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.fill_between(t, np.min(P_gt, axis=0), np.max(P_gt, axis=0),
                    color='gray', alpha=0.15, label='Simulator range')
    ax.plot(t, np.mean(P_gt, axis=0), 'k-', lw=2, label='Simulator (avg)')
    ax.plot(t, np.mean(P_v2, axis=0), color='steelblue', lw=2,
            ls='--', label='ST-GNN v2 (avg)')
    ax.fill_between(t, np.min(P_v2, axis=0), np.max(P_v2, axis=0),
                    color='steelblue', alpha=0.08, label='v2 range')

    phase_spans = [(0, 30, 'green'), (30, 43, 'gray'), (43, 73, '#e07000'),
                   (73, 116, 'green'), (116, 145, '#e07000')]
    for s, e, col in phase_spans:
        ax.axvspan(s, e, color=col, alpha=0.06)

    ax.set_xlabel("Timestep (5-day intervals)", fontsize=12)
    ax.set_ylabel("Pressure (bar)", fontsize=12)
    ax.set_title("Pressure Field Tracking: ST-GNN v2 vs Reservoir Simulator\n"
                 "(Case 5, shading = field min/max range)", fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(plot_dir / "fig10_pressure_tracking.png", dpi=600, bbox_inches='tight')
    fig.savefig(plot_dir / "fig10_pressure_tracking.pdf", bbox_inches='tight')
    plt.close(fig)
    print("  Saved -> fig10_pressure_tracking.png")


# ---------------------------------------------------------------------------
# Figure 4: Saturation tracking (v2 vs simulator)
# ---------------------------------------------------------------------------

def fig_saturation_tracking(plot_dir, Sg_gt, Sg_v2, n_steps):
    t = np.arange(n_steps)
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.fill_between(t, np.min(Sg_gt, axis=0), np.max(Sg_gt, axis=0),
                    color='gray', alpha=0.15, label='Simulator range')
    ax.plot(t, np.mean(Sg_gt, axis=0), 'k-', lw=2, label='Simulator (avg)')
    ax.plot(t, np.mean(Sg_v2, axis=0), color='steelblue', lw=2,
            ls='--', label='ST-GNN v2 (avg)')
    ax.fill_between(t, np.min(Sg_v2, axis=0), np.max(Sg_v2, axis=0),
                    color='steelblue', alpha=0.08, label='v2 range')

    phase_spans = [(0, 30, 'green'), (30, 43, 'gray'), (43, 73, '#e07000'),
                   (73, 116, 'green'), (116, 145, '#e07000')]
    for s, e, col in phase_spans:
        ax.axvspan(s, e, color=col, alpha=0.06)

    ax.set_xlabel("Timestep (5-day intervals)", fontsize=12)
    ax.set_ylabel("Gas Saturation (Sg)", fontsize=12)
    ax.set_title("Gas Saturation Tracking: ST-GNN v2 vs Reservoir Simulator\n"
                 "(Case 5, shading = field min/max range)", fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(plot_dir / "fig11_saturation_tracking.png", dpi=600, bbox_inches='tight')
    fig.savefig(plot_dir / "fig11_saturation_tracking.pdf", bbox_inches='tight')
    plt.close(fig)
    print("  Saved -> fig11_saturation_tracking.png")


# ---------------------------------------------------------------------------
# Figure 5: v1 vs v2 vs v3 bar comparison (multi-step)
# ---------------------------------------------------------------------------

def fig_bar_comparison(plot_dir, multi_step_data):
    step_labels = [1, 10, 50, 146]
    models = list(multi_step_data.keys())
    colors = {'v1': 'tomato', 'v2': 'steelblue', 'v3': 'darkorange'}

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle("Multi-Step Rollout Error Checkpoint Comparison: v1 vs v2 vs v3",
                 fontsize=16, fontweight='bold', y=0.96)

    x = np.arange(len(step_labels))
    width = 0.22

    for idx, (metric, ax, ylabel, title) in enumerate([
        ('P', axes[0], 'Pressure RMSE (bar)', 'Pressure Rollout Error ($\\Delta P$) at Checkpoints'),
        ('Sg', axes[1], 'Gas Saturation RMSE', 'Gas Saturation Rollout Error ($S_g$) at Checkpoints'),
    ]):
        max_val = 0.0
        for i, m in enumerate(models):
            vals = [multi_step_data[m].get(s, {}).get(metric, 0.0) for s in step_labels]
            max_val = max(max_val, max(vals) if vals else 0.0)
            
            bars = ax.bar(x + i * width, vals, width, label=f"ST-GNN {m}",
                          color=colors.get(m, 'gray'), alpha=0.85, edgecolor='black', lw=1.2)
            
            # Add readable value text on top of the bars
            for bar, v in zip(bars, vals):
                if metric == 'P':
                    txt = f"{v:.1f}"
                else:
                    txt = f"{v:.4f}"
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * max_val,
                        txt, ha='center', va='bottom', fontsize=9, fontweight='bold')

        # Thicker axes and styling
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=11, width=1.5, length=6)
        
        ax.set_xticks(x + width)
        ax.set_xticklabels([f"t = {s} steps" for s in step_labels], fontsize=11, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.legend(fontsize=11, loc='upper left')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # Headroom padding to prevent overlap with legends/boundaries
        ax.set_ylim(0, max_val * 1.25)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(plot_dir / "fig12_bar_comparison.png", dpi=600, bbox_inches='tight')
    fig.savefig(plot_dir / "fig12_bar_comparison.pdf", bbox_inches='tight')
    plt.close(fig)
    print("  Saved -> fig12_bar_comparison.png & .pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("PHASE 6v3: COMPREHENSIVE ROLLOUT EVALUATION")
    print("ST-GNN v1 vs v2 vs v3")
    print("=" * 80)

    project_root = Path(__file__).resolve().parents[2]
    data_raw_dir = project_root / "data" / "raw"
    data_processed_dir = project_root / "data" / "processed"
    results_figures_dir = project_root / "results" / "figures"
    results_metrics_dir = project_root / "results" / "metrics"
    
    # Make sure we also have the paper_project/figures path so Overleaf matches
    paper_figs_dir = project_root / "paper_project" / "figures"
    
    results_figures_dir.mkdir(parents=True, exist_ok=True)
    results_metrics_dir.mkdir(parents=True, exist_ok=True)
    paper_figs_dir.mkdir(parents=True, exist_ok=True)
    
    data_dir = data_processed_dir # read stats from data/processed
    plot_dir = results_figures_dir

    # Check required files
    required = {
        'v1': (project_root / 'checkpoints/stgnn_v1/st_gnn_checkpoint.pt',    'st_gnn_stats.json',    'ST_GNN'),
        'v2': (project_root / 'checkpoints/stgnn_v2/st_gnn_v2_checkpoint.pt', 'st_gnn_v2_stats.json', 'ST_GNN_v2'),
        'v3': (project_root / 'checkpoints/stgnn_v3/st_gnn_v3_checkpoint.pt', 'st_gnn_v3_stats.json', 'ST_GNN_v2'),
    }
    available = {}
    for name, (ckpt, stats, arch) in required.items():
        if Path(ckpt).exists() and (data_processed_dir / stats).exists():
            available[name] = (Path(ckpt), stats, arch)
            print(f"  [OK] {name}: {Path(ckpt).name}")
        else:
            print(f"  [MISSING] {name}: {Path(ckpt).name} or {stats} not found -- skipping")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Load shared data
    graph_data = np.load(data_processed_dir / "mesh_graph.npz")
    case_data  = sio.loadmat(str(data_raw_dir / "Case_0005_wj.mat"))
    n_cells, n_steps = case_data['P_matrix'].shape
    P_gt  = case_data['P_matrix'].astype(np.float32)
    Sg_gt = case_data['Sg_matrix'].astype(np.float32)

    # Load and run rollouts
    rollout_data = {}
    multi_step   = {}

    for name, (ckpt, stats_file, arch) in available.items():
        with open(data_dir / stats_file) as f:
            stats = json.load(f)
        mean_ = np.array(stats['feature_mean'], dtype=np.float32)
        std_  = np.array(stats['feature_std'],  dtype=np.float32)
        dp_std  = stats['target_dp_std']
        dsg_std = stats['target_dsg_std']

        if arch == 'ST_GNN':
            model = ST_GNN(in_features=len(mean_), hidden_features=64, out_features=2)
        else:
            model = ST_GNN_v2(in_features=len(mean_), hidden_features=128, out_features=2,
                              dropout=0.0)

        model.load_state_dict(torch.load(ckpt, map_location=device))
        model = model.to(device)

        P_pred, Sg_pred, rmse_P, rmse_Sg = run_rollout(
            model, device, case_data, graph_data,
            mean_, std_, dp_std, dsg_std, label=name)

        rollout_data[name] = {'rmse_P': rmse_P, 'rmse_Sg': rmse_Sg,
                              'P_pred': P_pred, 'Sg_pred': Sg_pred}
        multi_step[name]   = multistep_summary(rmse_P, rmse_Sg)

    # -----------------------------------------------------------------------
    # Print multi-step summary table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("MULTI-STEP ROLLOUT RMSE SUMMARY")
    print("=" * 70)
    print(f"\n  {'Model':<8} {'1-step P':>10} {'10-step P':>10} "
          f"{'50-step P':>10} {'146-step P':>11}")
    print("  " + "-" * 50)
    for name in available:
        ms = multi_step[name]
        print(f"  {name:<8} "
              f"{ms[1]['P']:>10.4f} {ms[10]['P']:>10.4f} "
              f"{ms[50]['P']:>10.4f} {ms[146]['P']:>11.4f}")

    print(f"\n  {'Model':<8} {'1-step Sg':>10} {'10-step Sg':>10} "
          f"{'50-step Sg':>10} {'146-step Sg':>11}")
    print("  " + "-" * 50)
    for name in available:
        ms = multi_step[name]
        print(f"  {name:<8} "
              f"{ms[1]['Sg']:>10.6f} {ms[10]['Sg']:>10.6f} "
              f"{ms[50]['Sg']:>10.6f} {ms[146]['Sg']:>11.6f}")

    # Also print mean and max error comparison
    print(f"\n  {'Model':<8} {'Mean P-RMSE':>12} {'Mean Sg-RMSE':>12} {'Max P-Error':>12} {'Max Sg-Error':>12}")
    print("  " + "-" * 65)
    for name in available:
        rd = rollout_data[name]
        max_p_err = float(np.max(np.abs(rd['P_pred'] - P_gt)))
        max_sg_err = float(np.max(np.abs(rd['Sg_pred'] - Sg_gt)))
        print(f"  {name:<8} {np.mean(rd['rmse_P']):>12.4f} {np.mean(rd['rmse_Sg']):>12.6f} {max_p_err:>12.4f} {max_sg_err:>12.6f}")
    print("=" * 70)

    # Save comprehensive results
    results_out = {
        name: {
            'multi_step': {str(k): v for k, v in multi_step[name].items()},
            'mean_P_rmse':  float(np.mean(rollout_data[name]['rmse_P'])),
            'mean_Sg_rmse': float(np.mean(rollout_data[name]['rmse_Sg'])),
            'max_P_error':  float(np.max(np.abs(rollout_data[name]['P_pred'] - P_gt))),
            'max_Sg_error': float(np.max(np.abs(rollout_data[name]['Sg_pred'] - Sg_gt))),
        }
        for name in available
    }
    with open(results_metrics_dir / 'v3_evaluation_results.json', 'w') as f:
        json.dump(results_out, f, indent=2)
    print("\n  Saved -> results/metrics/v3_evaluation_results.json")

    # -----------------------------------------------------------------------
    # Generate all 5 figures
    # -----------------------------------------------------------------------
    print("\nGenerating publication figures...")

    fig_training_curves(plot_dir, data_dir)

    fig_rollout_rmse(plot_dir, {n: rollout_data[n] for n in available})

    if 'v2' in rollout_data:
        fig_pressure_tracking(plot_dir, P_gt, rollout_data['v2']['P_pred'], n_steps)
        fig_saturation_tracking(plot_dir, Sg_gt, rollout_data['v2']['Sg_pred'], n_steps)
    elif 'v3' in rollout_data:
        fig_pressure_tracking(plot_dir, P_gt, rollout_data['v3']['P_pred'], n_steps)
        fig_saturation_tracking(plot_dir, Sg_gt, rollout_data['v3']['Sg_pred'], n_steps)

    fig_bar_comparison(plot_dir, multi_step)

    
    # Copy generated figures to paper_project/figures/ with final paper numbering
    import shutil
    paper_figs_dir = project_root / "paper_project" / "figures"
    paper_figs_dir.mkdir(exist_ok=True)
    mappings = [
        ("fig8_training_curves", "fig8_training_curves"),
        ("fig9_rollout_rmse", "fig9_rollout_rmse"),
        ("fig10_pressure_tracking", "fig10_pressure_tracking"),
        ("fig11_saturation_tracking", "fig11_saturation_tracking"),
        ("fig12_bar_comparison", "fig12_bar_comparison"),
    ]
    for src_base, dest_base in mappings:
        for ext in [".png", ".pdf"]:
            src_file = results_figures_dir / f"{src_base}{ext}"
            dest_file = paper_figs_dir / f"{dest_base}{ext}"
            if src_file.exists():
                shutil.copy2(src_file, dest_file)
    print("  Copied rollout figures to paper_project/figures/ with paper mapping.")

    print("\n[DONE] All figures saved to results/figures/ and paper_project/figures/")
    print("=" * 80)


if __name__ == "__main__":
    main()
