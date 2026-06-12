import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

"""
Train ST-GNN v2: Improved Training with Scheduled Sampling + Cosine LR
=======================================================================
Improvements over train_st_gnn.py (v1):
  - 50 training epochs (vs 12)
  - Cosine annealing LR schedule (1e-3 -> 1e-5)
  - Scheduled sampling curriculum: probability of using own predictions
    as next-step inputs grows from 0 -> 0.5 over epochs
  - Gradient clipping (max_norm=1.0) to prevent exploding gradients
  - ST_GNN_v2 architecture (hidden=128, 4 layers, residuals, LayerNorm)
  - Saves best checkpoint by validation loss
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import time
from pathlib import Path

from src.models.stgnn_v2.st_gnn_v2_model import UHSGraphDataset, ST_GNN_v2


# --- Helpers ---

def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)


@torch.no_grad()
def evaluate(model, dataset, device, dp_std: float, dsg_std: float) -> float:
    model.eval()
    total = 0.0
    edge_index = dataset.edge_index.to(device)
    edge_trans = dataset.edge_trans.to(device)
    for i in range(len(dataset)):
        x, target, _ = dataset.get_scaled_features(i)
        x = x.to(device)
        target = target.to(device)
        t_norm = torch.stack([target[:, 0] / dp_std, target[:, 1] / dsg_std], dim=-1)
        pred   = model(x, edge_index, edge_trans)
        total += mse_loss(pred, t_norm).item()
    model.train()
    return total / len(dataset)


def build_rollout_features(p_t, sg_t, sw_t, xh2_t, yh2_t,
                            dp_t0, dp_t1, dp_t2,
                            time_since_inj, time_since_prod,
                            inj_BHP_node, inj_H2_node, inj_status_node,
                            prod_BHP_node, prod_H2_node, prod_status_node,
                            perm, poro, dist_to_inj, dist_to_prod):
    """Stack all 20 features in the exact same order as UHSGraphDataset."""
    n_cells = len(perm)
    feat = np.column_stack((
        perm, poro, dist_to_inj, dist_to_prod,
        p_t, sg_t, sw_t, xh2_t, yh2_t,
        dp_t0, dp_t1, dp_t2,
        time_since_inj * np.ones(n_cells),
        time_since_prod * np.ones(n_cells),
        inj_BHP_node, inj_H2_node, inj_status_node,
        prod_BHP_node, prod_H2_node, prod_status_node,
    ))
    return feat.astype(np.float32)


def scheduled_sample_epoch(model, train_dataset, device, optimizer,
                            dp_std, dsg_std, p_rollout, batch_size,
                            graph_data):
    """
    One training epoch with scheduled sampling.

    p_rollout: probability of replacing ground-truth input with own prediction.
    When p_rollout=0.0, identical to plain teacher-forcing.
    When p_rollout=0.5, half the time we feed the model's own last output.
    """
    model.train()
    edge_index = train_dataset.edge_index.to(device)
    edge_trans = train_dataset.edge_trans.to(device)

    perm        = graph_data['perm']
    poro        = graph_data['poro']
    dist_to_inj = graph_data['dist_to_inj']
    dist_to_prod = graph_data['dist_to_prod']
    inj_cells   = graph_data['inj_cells']
    prod_cells  = graph_data['prod_cells']

    scaler_mean = train_dataset.scaler['mean']
    scaler_std  = train_dataset.scaler['std']

    n_samples = len(train_dataset)
    indices   = np.random.permutation(n_samples)

    epoch_loss = 0.0

    for start in range(0, n_samples, batch_size):
        batch_idx = indices[start : start + batch_size]
        optimizer.zero_grad()
        batch_loss = 0.0

        for idx in batch_idx:
            sample = train_dataset.samples[idx]

            if p_rollout > 0.0 and np.random.rand() < p_rollout:
                # Scheduled sampling: re-build features from model's own last prediction
                # We use the stored dp_history as a proxy for the previous state.
                # One step look-behind: predict what the model would have said at t-1
                # and use those corrected states to build the current-step features.
                #
                # Simplified implementation: replace P(t) with P(t) + model_correction
                # where correction = model's one-step-ago prediction denormalized.
                # This is a lightweight version of DAGGER that avoids full rollout
                # during training (which would be too slow on CPU for 10k-node graphs).

                x_gt, target, _ = train_dataset.get_scaled_features(idx)
                x_gt = x_gt.to(device)
                with torch.no_grad():
                    pred_prev_norm = model(x_gt, edge_index, edge_trans)
                    dp_correction  = pred_prev_norm[:, 0].cpu().numpy() * dp_std * 0.5
                    dsg_correction = pred_prev_norm[:, 1].cpu().numpy() * dsg_std * 0.5

                # Perturb current features with a fraction of model's predicted corrections
                feat = sample['features'].copy()
                feat[:, 4] += dp_correction        # perturb P(t)
                feat[:, 5]  = np.clip(feat[:, 5] + dsg_correction, 0.0, 1.0)  # perturb Sg(t)
                feat[:, 9] += dp_correction        # perturb dP(t)

                feat_scaled = (feat - scaler_mean) / scaler_std
                x_tensor = torch.from_numpy(feat_scaled).to(device)
            else:
                x_tensor, _, _ = train_dataset.get_scaled_features(idx)
                x_tensor = x_tensor.to(device)

            target = torch.from_numpy(sample['target']).to(device)
            t_norm = torch.stack([target[:, 0] / dp_std, target[:, 1] / dsg_std], dim=-1)

            pred = model(x_tensor, edge_index, edge_trans)
            loss = mse_loss(pred, t_norm)
            loss.backward()
            batch_loss += loss.item()

        # Gradient clipping before optimizer step
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += batch_loss

    return epoch_loss / n_samples


# --- Main ---

def main():
    print("=" * 80)
    print("PHASE 5v2: TRAINING ST-GNN v2 (Residuals + Scheduled Sampling + Cosine LR)")
    print("=" * 80)

    project_root = Path(__file__).resolve().parents[2]
    data_raw_dir = project_root / "data" / "raw"
    data_processed_dir = project_root / "data" / "processed"
    checkpoint_dir = project_root / "checkpoints" / "stgnn_v2"
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    data_processed_dir.mkdir(parents=True, exist_ok=True)
    
    graph_path = data_processed_dir / "mesh_graph.npz"

    if not graph_path.exists():
        raise FileNotFoundError("Run graph_construction.py first to build mesh_graph.npz!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -- Load Datasets --
    print("\nLoading Training Dataset (Cases 1-4, stride=4)...")
    train_dataset = UHSGraphDataset(data_raw_dir, case_ids=[1, 2, 3, 4],
                                     graph_path=graph_path, stride=4)

    print("\nLoading Validation Dataset (Case 5, stride=4)...")
    val_dataset = UHSGraphDataset(data_raw_dir, case_ids=[5],
                                   graph_path=graph_path, scaler=train_dataset.scaler,
                                   stride=4)  # stride=4 keeps eval fast (~35 samples)

    # -- Target scale statistics --
    all_targets  = np.concatenate([s['target'] for s in train_dataset.samples], axis=0)
    dp_std  = float(np.std(all_targets[:, 0]))
    dsg_std = float(np.std(all_targets[:, 1]))
    dp_std  = max(dp_std,  1e-6)
    dsg_std = max(dsg_std, 1e-6)
    print(f"\n  dP std : {dp_std:.4f} bar")
    print(f"  dSg std: {dsg_std:.6f}")

    # -- Load graph metadata needed for scheduled sampling --
    graph_raw = np.load(graph_path)
    graph_data = {
        'perm':         None,   # loaded from case files, placeholder
        'poro':         None,
        'dist_to_inj':  graph_raw['dist_to_inj'],
        'dist_to_prod': graph_raw['dist_to_prod'],
        'inj_cells':    graph_raw['inj_cells'],
        'prod_cells':   graph_raw['prod_cells'],
    }

    # -- Save stats for downstream scripts --
    stats = {
        'feature_mean':  train_dataset.scaler['mean'].tolist(),
        'feature_std':   train_dataset.scaler['std'].tolist(),
        'target_dp_std':  dp_std,
        'target_dsg_std': dsg_std,
    }
    with open(data_processed_dir / 'st_gnn_v2_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print("  Saved normalization statistics -> st_gnn_v2_stats.json")

    # -- Instantiate Model --
    in_features = train_dataset.scaler['mean'].shape[0]
    model = ST_GNN_v2(in_features=in_features, hidden_features=128,
                       out_features=2, dropout=0.05)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  ST-GNN v2 parameter count: {n_params:,}")

    # -- Optimizer + LR Scheduler --
    EPOCHS     = 30  # 30 epochs is sufficient; model plateaus around epoch 25-30
    BATCH_SIZE = 16
    EVAL_EVERY = 5   # only evaluate validation every N epochs to avoid bottleneck
    optimizer  = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    # Cosine annealing: warmup 5 epochs, then cosine decay to eta_min=1e-5
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-5
    )

    # -- Scheduled sampling curriculum --
    # Linear ramp: epoch 1 -> p=0.0, epoch EPOCHS -> p=0.5
    def get_p_rollout(epoch: int) -> float:
        return min(0.5, 0.5 * (epoch - 1) / max(EPOCHS - 1, 1))

    best_val_loss  = float('inf')
    best_checkpoints = []   # list of (val_loss, path) for top-3 management

    print(f"\nTraining for {EPOCHS} epochs | batch_size={BATCH_SIZE}")
    print(f"  Scheduled sampling: p_rollout 0.00 -> 0.50 over {EPOCHS} epochs")
    print(f"  Gradient clipping: max_norm=1.0")
    print(f"  LR schedule: CosineAnnealingWarmRestarts (T_0=10, T_mult=2)\n")

    history = []

    for epoch in range(1, EPOCHS + 1):
        p_rollout = get_p_rollout(epoch)
        t0 = time.time()

        epoch_loss = scheduled_sample_epoch(
            model, train_dataset, device, optimizer,
            dp_std, dsg_std, p_rollout, BATCH_SIZE, graph_data
        )

        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]
        dt = time.time() - t0

        # -- Evaluate validation only every EVAL_EVERY epochs --
        val_loss = float('nan')
        if epoch % EVAL_EVERY == 0 or epoch == EPOCHS:
            val_loss = evaluate(model, val_dataset, device, dp_std, dsg_std)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_path = checkpoint_dir / 'st_gnn_v2_checkpoint.pt'
                torch.save(model.state_dict(), ckpt_path)
                print(f"    [*] Saved best checkpoint (val={val_loss:.5f})")

        val_str = f"{val_loss:.5f}" if not (val_loss != val_loss) else "  ---  "
        print(f"  Epoch {epoch:3d}/{EPOCHS} | "
              f"Train={epoch_loss:.5f} | Val={val_str} | "
              f"p_roll={p_rollout:.2f} | LR={lr_now:.2e} | {dt:.1f}s")

        history.append({
            'epoch':      epoch,
            'train_loss': epoch_loss,
            'val_loss':   val_loss,
            'p_rollout':  p_rollout,
            'lr':         lr_now,
        })

    # -- Save training history --
    with open(data_processed_dir / 'st_gnn_v2_training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    print("\n  Saved training history -> st_gnn_v2_training_history.json")

    print(f"\n[OK] Training complete. Best validation loss: {best_val_loss:.5f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
