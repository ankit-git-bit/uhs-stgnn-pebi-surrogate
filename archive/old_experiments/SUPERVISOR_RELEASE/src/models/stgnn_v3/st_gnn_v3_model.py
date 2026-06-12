"""
ST-GNN v2: Improved Spatio-Temporal Graph Neural Network for UHS Surrogate Modeling
====================================================================================
Key improvements over v1:
  - Residual (skip) connections between DarcyGraphConv layers
  - LayerNorm after each message-passing step (stable training on graphs)
  - Wider network: hidden_features = 128 (vs 64 in v1)
  - 4 message-passing layers (vs 3 in v1)
  - Deeper MLP decoder with dropout regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
from pathlib import Path


# ---------------------------------------------------------------------------
# Graph Convolution Layer with Residual Connection and LayerNorm
# ---------------------------------------------------------------------------

class DarcyGraphConvResidual(nn.Module):
    """
    Darcy-weighted Graph Convolution with:
      - Transmissibility-weighted message aggregation (same physics as v1)
      - Residual (skip) projection: out = LayerNorm(conv(x) + proj(x))
      - LayerNorm applied AFTER residual addition for training stability

    This prevents gradient vanishing through deep stacks and allows the model
    to learn incremental corrections rather than full state transforms.
    """

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.0):
        super().__init__()
        self.linear_self = nn.Linear(in_features, out_features)
        self.linear_neigh = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(p=dropout)

        # Residual projection: needed when in_features != out_features
        if in_features != out_features:
            self.residual_proj = nn.Linear(in_features, out_features, bias=False)
        else:
            self.residual_proj = nn.Identity()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x           : Node features (N, in_features)
            edge_index  : COO adjacency (2, E)
            edge_weight : Transmissibilities (E,)
        Returns:
            out         : Updated node features (N, out_features)
        """
        row, col = edge_index[0], edge_index[1]
        num_nodes = x.size(0)

        # -- Self projection ------------------------------------------------
        x_self = self.linear_self(x)          # (N, out_features)

        # -- Neighbour projection -------------------------------------------
        x_neigh = self.linear_neigh(x)        # (N, out_features)

        # -- Transmissibility normalisation (row-sum) -----------------------
        sum_w = torch.zeros(num_nodes, device=x.device)
        sum_w.index_add_(0, row, edge_weight)
        sum_w = torch.clamp(sum_w, min=1e-12)
        norm_w = edge_weight / sum_w[row]     # (E,)

        # -- Weighted aggregation -------------------------------------------
        weighted_msg = x_neigh[col] * norm_w.unsqueeze(-1)   # (E, out_features)
        aggregated = torch.zeros_like(x_self)
        aggregated.index_add_(0, row, weighted_msg)           # (N, out_features)

        # -- Conv output + residual + LayerNorm ----------------------------
        conv_out = F.leaky_relu(x_self + aggregated, negative_slope=0.1)
        residual = self.residual_proj(x)
        out = self.norm(conv_out + residual)
        out = self.dropout(out)
        return out


# ---------------------------------------------------------------------------
# ST-GNN v2 Model
# ---------------------------------------------------------------------------

class ST_GNN_v2(nn.Module):
    """
    ST-GNN v2: 4-layer DarcyGraphConvResidual stack with:
      - 128-dimensional hidden state
      - LayerNorm + residual connections in every layer
      - Deeper MLP decoder (4 fully-connected layers + dropout)
      - Output: [delta_P, delta_Sg]  (pressure and saturation changes)
    """

    def __init__(self, in_features: int, hidden_features: int = 128, out_features: int = 2, dropout: float = 0.05):
        super().__init__()

        # -- Input embedding ------------------------------------------------
        self.input_embed = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.LeakyReLU(0.1),
        )

        # -- 4 residual graph conv layers ----------------------------------
        self.conv1 = DarcyGraphConvResidual(hidden_features, hidden_features, dropout=dropout)
        self.conv2 = DarcyGraphConvResidual(hidden_features, hidden_features, dropout=dropout)
        self.conv3 = DarcyGraphConvResidual(hidden_features, hidden_features, dropout=dropout)
        self.conv4 = DarcyGraphConvResidual(hidden_features, hidden_features, dropout=dropout)

        # -- Global skip: concat input embedding with deep features ---------
        # After 4 conv layers, concatenate with the input embedding and decode
        self.decoder = nn.Sequential(
            nn.Linear(hidden_features * 2, hidden_features),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_features, hidden_features // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_features // 2, out_features),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x           : Raw node features (N, in_features)
            edge_index  : COO adjacency (2, E)
            edge_weight : Transmissibilities (E,)
        Returns:
            out         : Predicted state changes (N, 2) = [dP_norm, dSg_norm]
        """
        h0 = self.input_embed(x)          # (N, hidden)

        h = self.conv1(h0, edge_index, edge_weight)
        h = self.conv2(h, edge_index, edge_weight)
        h = self.conv3(h, edge_index, edge_weight)
        h = self.conv4(h, edge_index, edge_weight)

        # Global skip concatenation: preserves input context through deep stack
        h_cat = torch.cat([h0, h], dim=-1)   # (N, 2*hidden)
        out = self.decoder(h_cat)             # (N, 2)
        return out


# ---------------------------------------------------------------------------
# Dataset (reused from v1 with no changes -- same feature engineering)
# ---------------------------------------------------------------------------

def load_case_mat(case_path: Path):
    """Load .mat case file, stripping MATLAB metadata keys."""
    data = sio.loadmat(str(case_path))
    return {k: v for k, v in data.items() if not k.startswith('__')}


class UHSGraphDataset:
    """
    Dataset loader identical to v1 UHSGraphDataset.
    Builds 20-feature node inputs per timestep:
      [0] Perm, [1] Poro, [2] dist_inj, [3] dist_prod
      [4] P(t), [5] Sg(t), [6] Sw(t), [7] xH2(t), [8] yH2(t)
      [9] dP(t), [10] dP(t-1), [11] dP(t-2)
      [12] time_since_inj, [13] time_since_prod
      [14] inj_BHP, [15] inj_H2, [16] inj_status
      [17] prod_BHP, [18] prod_H2, [19] prod_status
    Targets: [delta_P, delta_Sg] at next timestep.
    """

    def __init__(self, data_dir: Path, case_ids: list, graph_path: Path, scaler=None, stride: int = 1):
        self.data_dir = Path(data_dir)
        self.case_files = [self.data_dir / f"Case_{cid:04d}_wj.mat" for cid in case_ids]
        self.stride = stride

        # -- Load static graph ---------------------------------------------
        graph_data = np.load(graph_path)
        self.edge_index  = torch.from_numpy(graph_data['edge_index'])
        self.edge_dist   = torch.from_numpy(graph_data['edge_dist'])
        self.edge_trans  = torch.from_numpy(graph_data['edge_trans'])
        self.dist_to_inj = graph_data['dist_to_inj']
        self.dist_to_prod = graph_data['dist_to_prod']
        self.inj_cells   = graph_data['inj_cells']
        self.prod_cells  = graph_data['prod_cells']
        self.num_nodes   = len(graph_data['X_coords'])
        self.scaler      = scaler

        self.samples = []
        self._load_and_process_cases()

        if self.scaler is None:
            self._fit_scaler()

    def _load_and_process_cases(self):
        for case_path in self.case_files:
            print(f"    Processing: {case_path.name}")
            data = load_case_mat(case_path)

            perm = data['Perm_initial'].squeeze()
            poro = data['Poro_initial'].squeeze()

            P   = data['P_matrix']
            Sg  = data['Sg_matrix']
            Sw  = data['Sw_matrix']
            xH2 = data['xH2_matrix']
            yH2 = data['yH2_matrix']

            inj_BHP    = data['inj_BHP'].squeeze()
            prod_BHP   = data['prod_BHP'].squeeze()
            inj_H2     = data['inj_H2'].squeeze()
            prod_H2    = data['prod_H2'].squeeze()
            inj_status  = data['inj_status'].squeeze()
            prod_status = data['prod_status'].squeeze()

            n_cells, n_steps = P.shape

            # Precompute elapsed time counters
            time_since_inj  = np.zeros(n_steps)
            time_since_prod = np.zeros(n_steps)
            last_inj_t = last_prod_t = 0
            for t in range(n_steps):
                if inj_status[t]  == 1: last_inj_t  = t
                if prod_status[t] == 1: last_prod_t = t
                time_since_inj[t]  = t - last_inj_t
                time_since_prod[t] = t - last_prod_t

            for t in range(3, n_steps - 1, self.stride):
                p_t   = P[:, t]
                sg_t  = Sg[:, t]
                sw_t  = Sw[:, t]
                xh2_t = xH2[:, t]
                yh2_t = yH2[:, t]

                dp_t0 = P[:, t]   - P[:, t-1]
                dp_t1 = P[:, t-1] - P[:, t-2]
                dp_t2 = P[:, t-2] - P[:, t-3]

                inj_BHP_node   = np.zeros(n_cells)
                inj_H2_node    = np.zeros(n_cells)
                inj_status_node = np.zeros(n_cells)
                prod_BHP_node   = np.zeros(n_cells)
                prod_H2_node    = np.zeros(n_cells)
                prod_status_node = np.zeros(n_cells)

                inj_BHP_node[self.inj_cells]    = inj_BHP[t]
                inj_H2_node[self.inj_cells]     = inj_H2[t]
                inj_status_node[self.inj_cells] = inj_status[t]
                prod_BHP_node[self.prod_cells]   = prod_BHP[t]
                prod_H2_node[self.prod_cells]    = prod_H2[t]
                prod_status_node[self.prod_cells] = prod_status[t]

                feat = np.column_stack((
                    perm, poro, self.dist_to_inj, self.dist_to_prod,
                    p_t, sg_t, sw_t, xh2_t, yh2_t,
                    dp_t0, dp_t1, dp_t2,
                    time_since_inj[t]  * np.ones(n_cells),
                    time_since_prod[t] * np.ones(n_cells),
                    inj_BHP_node, inj_H2_node, inj_status_node,
                    prod_BHP_node, prod_H2_node, prod_status_node,
                ))

                target_dp  = P[:, t+1]  - P[:, t]
                target_dsg = Sg[:, t+1] - Sg[:, t]
                target = np.column_stack((target_dp, target_dsg))

                self.samples.append({
                    'features': feat.astype(np.float32),
                    'target':   target.astype(np.float32),
                    'P_t':  p_t.astype(np.float32),
                    'Sg_t': sg_t.astype(np.float32),
                    'dp_history': np.column_stack((dp_t0, dp_t1, dp_t2)).astype(np.float32),
                    'timestep': t,
                    'case': case_path.name,
                })

    def _fit_scaler(self):
        print("    Computing normalization statistics...")
        all_feats = np.concatenate([s['features'] for s in self.samples], axis=0)
        means = np.mean(all_feats, axis=0)
        stds  = np.std(all_feats,  axis=0)
        stds[stds < 1e-8] = 1.0
        self.scaler = {'mean': means, 'std': stds}

    def get_scaled_features(self, index: int):
        sample = self.samples[index]
        feats = (sample['features'] - self.scaler['mean']) / self.scaler['std']
        return torch.from_numpy(feats), torch.from_numpy(sample['target']), sample

    def __len__(self):
        return len(self.samples)
