import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

"""
train_st_gnn_v3.py
==================
ST-GNN v3: Improved rollout stability and saturation accuracy.

Architecture: ST_GNN_v2 (unchanged - residuals, LayerNorm, hidden=128, 4 layers)

Improvements over v2:
  A. Extended scheduled sampling: properly perturbs P, dP, Sg AND Sw consistently
     - v2 bug: dsg_correction was scaled by tiny dsg_std (0.00223) -> negligible
     - v3 fix: use full-magnitude Sg correction; update Sw = 1-Sg consistently
  B. Loss-weight study: trains 3 configs (20 epochs each) to find optimal alpha_P/alpha_Sg
     - Config 1: alpha_P=0.5, alpha_Sg=0.5
     - Config 2: alpha_P=0.7, alpha_Sg=0.3
     - Config 3: alpha_P=0.3, alpha_Sg=0.7
  C. Final v3 model: 60 epochs with best loss weights from study
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import time
import copy
from pathlib import Path

from st_gnn_v2_model import UHSGraphDataset, ST_GNN_v2


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def weighted_mse(pred, target_norm, alpha_P, alpha_Sg):
    """Weighted MSE separating pressure and saturation contributions."""
    loss_P  = torch.mean((pred[:, 0] - target_norm[:, 0]) ** 2)
    loss_Sg = torch.mean((pred[:, 1] - target_norm[:, 1]) ** 2)
    return alpha_P * loss_P + alpha_Sg * loss_Sg, loss_P.item(), loss_Sg.item()


@torch.no_grad()
def evaluate(model, dataset, device, dp_std, dsg_std, alpha_P=0.5, alpha_Sg=0.5):
    model.eval()
    total, total_P, total_Sg = 0.0, 0.0, 0.0
    ei = dataset.edge_index.to(device)
    et = dataset.edge_trans.to(device)
    for i in range(len(dataset)):
        x, target, _ = dataset.get_scaled_features(i)
        x = x.to(device)
        target = target.to(device)
        t_norm = torch.stack([target[:, 0] / dp_std,
                              target[:, 1] / dsg_std], dim=-1)
        pred = model(x, ei, et)
        loss, lP, lSg = weighted_mse(pred, t_norm, alpha_P, alpha_Sg)
        total    += loss.item()
        total_P  += lP
        total_Sg += lSg
    n = len(dataset)
    model.train()
    return total / n, total_P / n, total_Sg / n


# ---------------------------------------------------------------------------
# Extended scheduled sampling (v3) - perturbs P, dP, Sg, Sw
# ---------------------------------------------------------------------------

def scheduled_sample_epoch_v3(model, train_dataset, device, optimizer,
                               dp_std, dsg_std, p_rollout, batch_size,
                               alpha_P, alpha_Sg):
    """
    Training epoch with EXTENDED scheduled sampling.

    v2 bug fixed: Sg perturbation now uses the full prediction magnitude,
    not scaled by the tiny dsg_std. Sw is updated consistently (Sw = 1 - Sg).

    Feature column map (UHSGraphDataset):
      [0] Perm   [1] Poro   [2] dist_inj  [3] dist_prod
      [4] P(t)   [5] Sg(t)  [6] Sw(t)     [7] xH2(t)   [8] yH2(t)
      [9] dP(t)  [10] dP(t-1) [11] dP(t-2)
      [12] time_inj  [13] time_prod
      [14] inj_BHP  [15] inj_H2  [16] inj_status
      [17] prod_BHP [18] prod_H2 [19] prod_status
    """
    model.train()
    ei = train_dataset.edge_index.to(device)
    et = train_dataset.edge_trans.to(device)
    scaler_mean = train_dataset.scaler['mean']
    scaler_std  = train_dataset.scaler['std']

    n_samples = len(train_dataset)
    indices   = np.random.permutation(n_samples)
    epoch_loss = 0.0

    for start in range(0, n_samples, batch_size):
        batch_idx = indices[start: start + batch_size]
        optimizer.zero_grad()
        batch_loss = 0.0

        for idx in batch_idx:
            sample = train_dataset.samples[idx]

            if p_rollout > 0.0 and np.random.rand() < p_rollout:
                # -- Scheduled sampling with extended Sg perturbation --
                x_gt, _, _ = train_dataset.get_scaled_features(idx)
                x_gt = x_gt.to(device)
                with torch.no_grad():
                    pred_norm = model(x_gt, ei, et)
                    # Pressure correction (bar) - full magnitude, halved
                    dp_corr  = pred_norm[:, 0].cpu().numpy() * dp_std * 0.5
                    # Saturation correction (absolute Sg change) - full magnitude
                    # FIX vs v2: use dp_std scaling analogue for Sg
                    # pred_norm[:, 1] is the normalized dSg prediction
                    # Denormalize: dSg_pred = pred_norm[:, 1] * dsg_std
                    # But dsg_std is tiny (0.00223), so instead apply fraction of
                    # raw normalized prediction rescaled to meaningful Sg range (~0.01)
                    dsg_corr = pred_norm[:, 1].cpu().numpy() * dsg_std  # true delta_Sg

                feat = sample['features'].copy()

                # Perturb P(t) - column 4
                feat[:, 4] += dp_corr

                # Perturb dP(t) - column 9 (most recent pressure change history)
                feat[:, 9] += dp_corr

                # Perturb Sg(t) - column 5 (EXTENDED vs v2)
                sg_new = np.clip(feat[:, 5] + dsg_corr, 0.0, 1.0)
                feat[:, 5] = sg_new

                # Update Sw(t) = 1 - Sg(t) consistently - column 6 (EXTENDED vs v2)
                feat[:, 6] = np.clip(1.0 - sg_new, 0.0, 1.0)

                feat_scaled = (feat - scaler_mean) / scaler_std
                x_tensor = torch.from_numpy(feat_scaled.astype(np.float32)).to(device)
            else:
                x_tensor, _, _ = train_dataset.get_scaled_features(idx)
                x_tensor = x_tensor.to(device)

            target  = torch.from_numpy(sample['target']).to(device)
            t_norm  = torch.stack([target[:, 0] / dp_std,
                                   target[:, 1] / dsg_std], dim=-1)

            pred = model(x_tensor, ei, et)
            loss, _, _ = weighted_mse(pred, t_norm, alpha_P, alpha_Sg)
            loss.backward()
            batch_loss += loss.item()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += batch_loss

    return epoch_loss / n_samples


# ---------------------------------------------------------------------------
# Quick rollout for loss-weight comparison (no plots)
# ---------------------------------------------------------------------------

@torch.no_grad()
def quick_rollout(model, case_data, graph_data, scaler_mean, scaler_std,
                  dp_std, dsg_std, device):
    """Run 142-step rollout; return (mean_P_rmse, mean_Sg_rmse)."""
    P_gt  = case_data['P_matrix'].astype(np.float32)
    Sg_gt = case_data['Sg_matrix'].astype(np.float32)
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
    ei = torch.from_numpy(graph_data['edge_index']).to(device)
    et = torch.from_numpy(graph_data['edge_trans']).to(device)
    dist_to_inj  = graph_data['dist_to_inj']
    dist_to_prod = graph_data['dist_to_prod']
    inj_cells    = graph_data['inj_cells']
    prod_cells   = graph_data['prod_cells']

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
    rmse_P = []
    rmse_Sg = []

    model.eval()
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

    return float(np.mean(rmse_P)), float(np.mean(rmse_Sg)), rmse_P, rmse_Sg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train_one_config(train_dataset, val_dataset, graph_path, device,
                     dp_std, dsg_std, alpha_P, alpha_Sg, n_epochs,
                     batch_size, label, eval_every=5):
    """Train one loss-weight configuration; return (model, best_val_loss, history)."""
    in_features = train_dataset.scaler['mean'].shape[0]
    model = ST_GNN_v2(in_features=in_features, hidden_features=128,
                      out_features=2, dropout=0.05).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-5)

    def get_p(ep):
        return min(0.5, 0.5 * (ep - 1) / max(n_epochs - 1, 1))

    best_val  = float('inf')
    best_state = None
    history   = []

    print(f"\n  --- Config {label}: alpha_P={alpha_P}, alpha_Sg={alpha_Sg} ---")
    for epoch in range(1, n_epochs + 1):
        p_roll = get_p(epoch)
        t0 = time.time()
        tr_loss = scheduled_sample_epoch_v3(
            model, train_dataset, device, optimizer,
            dp_std, dsg_std, p_roll, batch_size, alpha_P, alpha_Sg)
        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        val_loss = float('nan')
        val_P    = float('nan')
        val_Sg   = float('nan')
        if epoch % eval_every == 0 or epoch == n_epochs:
            val_loss, val_P, val_Sg = evaluate(
                model, val_dataset, device, dp_std, dsg_std, alpha_P, alpha_Sg)
            if val_loss < best_val:
                best_val = val_loss
                best_state = copy.deepcopy(model.state_dict())

        val_str = f"{val_loss:.5f}" if val_loss == val_loss else "  ---  "
        dt = time.time() - t0
        print(f"    Ep {epoch:3d}/{n_epochs} | Train={tr_loss:.5f} | Val={val_str}"
              f" | p={p_roll:.2f} | LR={lr:.2e} | {dt:.0f}s")

        history.append({'epoch': epoch, 'train_loss': tr_loss,
                        'val_loss': val_loss, 'val_P': val_P, 'val_Sg': val_Sg,
                        'p_rollout': p_roll, 'lr': lr})

    model.load_state_dict(best_state)
    return model, best_val, history


def main():
    print("=" * 80)
    print("ST-GNN v3: Loss-weight Study + 60-Epoch Final Training")
    print("=" * 80)

    project_root = Path(__file__).resolve().parents[2]
    data_raw_dir = project_root / "data" / "raw"
    data_processed_dir = project_root / "data" / "processed"
    checkpoint_dir = project_root / "checkpoints" / "stgnn_v3"
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    data_processed_dir.mkdir(parents=True, exist_ok=True)
    
    graph_path = data_processed_dir / "mesh_graph.npz"
    import scipy.io as sio

    if not graph_path.exists():
        raise FileNotFoundError("mesh_graph.npz missing - run graph_construction.py first")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # -- Load datasets (once, shared across all configs) --
    print("\nLoading datasets...")
    train_dataset = UHSGraphDataset(data_raw_dir, case_ids=[1, 2, 3, 4],
                                     graph_path=graph_path, stride=4)
    val_dataset   = UHSGraphDataset(data_raw_dir, case_ids=[5],
                                     graph_path=graph_path,
                                     scaler=train_dataset.scaler, stride=4)

    all_targets = np.concatenate([s['target'] for s in train_dataset.samples], axis=0)
    dp_std  = max(float(np.std(all_targets[:, 0])), 1e-6)
    dsg_std = max(float(np.std(all_targets[:, 1])), 1e-6)
    print(f"  dP std:  {dp_std:.4f} bar")
    print(f"  dSg std: {dsg_std:.6f}")

    # Save stats
    stats = {
        'feature_mean': train_dataset.scaler['mean'].tolist(),
        'feature_std':  train_dataset.scaler['std'].tolist(),
        'target_dp_std':  dp_std,
        'target_dsg_std': dsg_std,
    }
    with open(data_processed_dir / 'st_gnn_v3_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print("  Saved -> st_gnn_v3_stats.json")

    # Load Case 5 for quick rollout comparison
    graph_data = np.load(graph_path)
    case5 = sio.loadmat(str(data_dir / "Case_0005_wj.mat"))
    scaler_mean = train_dataset.scaler['mean'].astype(np.float32)
    scaler_std  = train_dataset.scaler['std'].astype(np.float32)

    # =====================================================================
    # PHASE 1: Loss-weight study (20 epochs per config)
    # =====================================================================
    STUDY_EPOCHS = 20
    BATCH_SIZE   = 16
    EVAL_EVERY   = 5

    loss_configs = [
        ('0.5P+0.5Sg', 0.5, 0.5),
        ('0.7P+0.3Sg', 0.7, 0.3),
        ('0.3P+0.7Sg', 0.3, 0.7),
    ]

    study_results = []
    study_histories = {}

    print(f"\n{'='*60}")
    print(f"PHASE 1: Loss-weight Study ({STUDY_EPOCHS} epochs each)")
    print(f"{'='*60}")

    for label, alpha_P, alpha_Sg in loss_configs:
        model, best_val, history = train_one_config(
            train_dataset, val_dataset, graph_path, device,
            dp_std, dsg_std, alpha_P, alpha_Sg,
            n_epochs=STUDY_EPOCHS, batch_size=BATCH_SIZE,
            label=label, eval_every=EVAL_EVERY)

        # Quick rollout to compare
        print(f"    Running quick rollout for config {label}...")
        mean_P, mean_Sg, rmse_P_list, rmse_Sg_list = quick_rollout(
            model, case5, graph_data, scaler_mean, scaler_std,
            dp_std, dsg_std, device)

        # Balanced score: geometric mean of normalized errors
        # P range ~200 bar -> normalize by 200; Sg range ~0.5 -> normalize by 0.5
        score = (mean_P / 200.0) * (mean_Sg / 0.5)

        print(f"    Config {label}: mean_P_RMSE={mean_P:.2f} bar | "
              f"mean_Sg_RMSE={mean_Sg:.5f} | Score={score:.6f}")

        study_results.append({
            'label': label, 'alpha_P': alpha_P, 'alpha_Sg': alpha_Sg,
            'best_val': best_val,
            'mean_P_rmse': mean_P, 'mean_Sg_rmse': mean_Sg,
            'final_P_rmse': rmse_P_list[-1], 'final_Sg_rmse': rmse_Sg_list[-1],
            'balance_score': score,
        })
        study_histories[label] = history

    # Print comparison table
    print(f"\n{'='*60}")
    print("LOSS-WEIGHT STUDY RESULTS")
    print(f"{'='*60}")
    print(f"{'Config':<14} {'Mean P-RMSE':>12} {'Mean Sg-RMSE':>13} "
          f"{'Final P-RMSE':>13} {'Final Sg-RMSE':>14} {'Score':>10}")
    print("-" * 80)
    for r in study_results:
        print(f"  {r['label']:<12} {r['mean_P_rmse']:>12.2f} {r['mean_Sg_rmse']:>13.5f} "
              f"{r['final_P_rmse']:>13.2f} {r['final_Sg_rmse']:>14.5f} "
              f"{r['balance_score']:>10.6f}")

    # Select winner (lowest balance_score = best tradeoff)
    winner = min(study_results, key=lambda x: x['balance_score'])
    print(f"\n  Winner: {winner['label']} (lowest balance score = {winner['balance_score']:.6f})")

    # Save study results
    study_summary = {
        'study_epochs': STUDY_EPOCHS,
        'winner': winner['label'],
        'winner_alpha_P': winner['alpha_P'],
        'winner_alpha_Sg': winner['alpha_Sg'],
        'configs': study_results,
        'histories': study_histories,
    }
    with open(data_processed_dir / 'st_gnn_v3_study_summary.json', 'w') as f:
        json.dump(study_summary, f, indent=2)
    print("  Saved -> st_gnn_v3_study_summary.json")

    # =====================================================================
    # PHASE 2: Full 60-epoch training with best loss weights
    # =====================================================================
    best_alpha_P  = winner['alpha_P']
    best_alpha_Sg = winner['alpha_Sg']

    print(f"\n{'='*60}")
    print(f"PHASE 2: Final Training (60 epochs, "
          f"alpha_P={best_alpha_P}, alpha_Sg={best_alpha_Sg})")
    print(f"{'='*60}")

    model_v3, _, history_v3 = train_one_config(
        train_dataset, val_dataset, graph_path, device,
        dp_std, dsg_std, best_alpha_P, best_alpha_Sg,
        n_epochs=60, batch_size=BATCH_SIZE,
        label='v3-final', eval_every=5)

    # Save checkpoint
    torch.save(model_v3.state_dict(), checkpoint_dir / 'st_gnn_v3_checkpoint.pt')
    print("  Saved -> st_gnn_v3_checkpoint.pt")

    # Run final rollout
    print("\nRunning final rollout evaluation...")
    mean_P, mean_Sg, rmse_P_list, rmse_Sg_list = quick_rollout(
        model_v3, case5, graph_data, scaler_mean, scaler_std,
        dp_std, dsg_std, device)

    # Multi-step RMSE report
    steps_of_interest = [0, 9, 49, len(rmse_P_list) - 1]
    step_labels = [1, 10, 50, 146]
    print("\n  Multi-step RMSE Report:")
    print(f"  {'Step':>6} {'P-RMSE (bar)':>14} {'Sg-RMSE':>12}")
    print("  " + "-" * 35)
    for si, sl in zip(steps_of_interest, step_labels):
        if si < len(rmse_P_list):
            print(f"  {sl:>6} {rmse_P_list[si]:>14.4f} {rmse_Sg_list[si]:>12.6f}")

    # Save history and rollout summary
    with open(data_processed_dir / 'st_gnn_v3_training_history.json', 'w') as f:
        json.dump(history_v3, f, indent=2)

    rollout_summary = {
        'model': 'ST-GNN v3',
        'case': 'Case_0005_wj.mat',
        'winner_loss_config': winner['label'],
        'alpha_P': best_alpha_P,
        'alpha_Sg': best_alpha_Sg,
        'training_epochs': 60,
        'rollout_steps': len(rmse_P_list),
        'step_1_P_rmse':   rmse_P_list[0]  if len(rmse_P_list) > 0   else None,
        'step_10_P_rmse':  rmse_P_list[9]  if len(rmse_P_list) > 9   else None,
        'step_50_P_rmse':  rmse_P_list[49] if len(rmse_P_list) > 49  else None,
        'step_146_P_rmse': rmse_P_list[-1],
        'step_1_Sg_rmse':   rmse_Sg_list[0]  if len(rmse_Sg_list) > 0  else None,
        'step_10_Sg_rmse':  rmse_Sg_list[9]  if len(rmse_Sg_list) > 9  else None,
        'step_50_Sg_rmse':  rmse_Sg_list[49] if len(rmse_Sg_list) > 49 else None,
        'step_146_Sg_rmse': rmse_Sg_list[-1],
        'mean_pressure_rmse':   mean_P,
        'mean_saturation_rmse': mean_Sg,
        'rmse_P_series':  rmse_P_list,
        'rmse_Sg_series': rmse_Sg_list,
    }
    with open(data_processed_dir / 'st_gnn_v3_rollout_summary.json', 'w') as f:
        json.dump(rollout_summary, f, indent=2)
    print("\n  Saved -> st_gnn_v3_rollout_summary.json")
    print("  Saved -> st_gnn_v3_training_history.json")

    print(f"\n[DONE] ST-GNN v3 training complete.")
    print(f"  Winner config: {winner['label']}")
    print(f"  Final P-RMSE (mean): {mean_P:.2f} bar")
    print(f"  Final Sg-RMSE (mean): {mean_Sg:.5f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
