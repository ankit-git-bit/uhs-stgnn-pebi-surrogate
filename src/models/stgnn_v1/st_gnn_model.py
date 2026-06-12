import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
from pathlib import Path

class DarcyGraphConv(nn.Module):
    """
    Custom Graph Convolution layer that weights message aggregation
    by physical connection transmissibilities.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear_self = nn.Linear(in_features, out_features)
        self.linear_neigh = nn.Linear(in_features, out_features)
        
    def forward(self, x, edge_index, edge_weight):
        """
        x: Node features of shape (num_nodes, in_features)
        edge_index: Adjacency list of shape (2, num_edges)
        edge_weight: Transmissibilities of shape (num_edges,)
        """
        # Self-features projection
        x_self = self.linear_self(x)
        
        # Neighbor-features projection
        x_neigh = self.linear_neigh(x)
        
        row, col = edge_index[0], edge_index[1]
        num_nodes = x.size(0)
        
        # Normalize edge weights (transmissibilities) per receiving node
        sum_weight = torch.zeros(num_nodes, device=x.device)
        sum_weight.index_add_(0, row, edge_weight)
        sum_weight = torch.clamp(sum_weight, min=1e-12)
        
        norm_weight = edge_weight / sum_weight[row]
        
        # Weighted message passing
        weighted_features = x_neigh[col] * norm_weight.unsqueeze(-1)
        aggregated = torch.zeros_like(x_self)
        aggregated.index_add_(0, row, weighted_features)
        
        # Combine self and aggregated neighbor features
        out = x_self + aggregated
        return F.leaky_relu(out, negative_slope=0.1)


class ST_GNN(nn.Module):
    """
    Spatio-Temporal Graph Neural Network for UHS simulation forecasting.
    Predicts pressure and saturation change: [delta_P, delta_Sg].
    """
    def __init__(self, in_features, hidden_features=64, out_features=2):
        super().__init__()
        self.conv1 = DarcyGraphConv(in_features, hidden_features)
        self.conv2 = DarcyGraphConv(hidden_features, hidden_features)
        self.conv3 = DarcyGraphConv(hidden_features, hidden_features)
        
        # Decoder mapping updated representations to target output changes
        self.decoder = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_features, hidden_features // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_features // 2, out_features)
        )
        
    def forward(self, x, edge_index, edge_weight):
        h = self.conv1(x, edge_index, edge_weight)
        h = self.conv2(h, edge_index, edge_weight)
        h = self.conv3(h, edge_index, edge_weight)
        out = self.decoder(h)
        return out


def load_case_mat(case_path: Path):
    """Utility to load case file using scipy."""
    data = sio.loadmat(str(case_path))
    return {k: v for k, v in data.items() if not k.startswith('__')}


class UHSGraphDataset:
    """
    Dataset loader preparing graph-compatible dynamic node features
    and dynamic targets for training/evaluation.
    """
    def __init__(self, data_dir: Path, case_ids: list, graph_path: Path, scaler=None, stride=1):
        self.data_dir = Path(data_dir)
        self.case_files = [self.data_dir / f"Case_{cid:04d}_wj.mat" for cid in case_ids]
        self.stride = stride
        
        # Load static graph representation
        graph_data = np.load(graph_path)
        self.edge_index = torch.from_numpy(graph_data['edge_index'])
        self.edge_dist = torch.from_numpy(graph_data['edge_dist'])
        self.edge_trans = torch.from_numpy(graph_data['edge_trans'])
        self.dist_to_inj = graph_data['dist_to_inj']
        self.dist_to_prod = graph_data['dist_to_prod']
        self.inj_cells = graph_data['inj_cells']
        self.prod_cells = graph_data['prod_cells']
        
        self.num_nodes = len(graph_data['X_coords'])
        self.scaler = scaler
        
        # Load and stack all cases
        self.samples = []
        self._load_and_process_cases()
        
        # Fit scaler if not provided
        if self.scaler is None:
            self._fit_scaler()
            
    def _load_and_process_cases(self):
        for case_idx, case_path in enumerate(self.case_files):
            print(f"    Processing case: {case_path.name}")
            data = load_case_mat(case_path)
            
            # Static attributes
            perm = data['Perm_initial'].squeeze()
            poro = data['Poro_initial'].squeeze()
            
            # Dynamic attributes (num_cells, num_timesteps)
            P = data['P_matrix']
            Sg = data['Sg_matrix']
            Sw = data['Sw_matrix']
            xH2 = data['xH2_matrix']
            yH2 = data['yH2_matrix']
            
            # Control time-series
            inj_BHP = data['inj_BHP'].squeeze()
            prod_BHP = data['prod_BHP'].squeeze()
            inj_H2 = data['inj_H2'].squeeze()
            prod_H2 = data['prod_H2'].squeeze()
            inj_status = data['inj_status'].squeeze()
            prod_status = data['prod_status'].squeeze()
            
            n_cells, n_steps = P.shape
            n_pairs = n_steps - 3 # We need at least 3 historical steps
            
            # Precompute dynamic elapsed time since injection/production cycle starts
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
                
            # Construct time samples
            for t in range(3, n_steps - 1, self.stride):
                # 1. State features at t
                p_t = P[:, t]
                sg_t = Sg[:, t]
                sw_t = Sw[:, t]
                xh2_t = xH2[:, t]
                yh2_t = yH2[:, t]
                
                # 2. History features
                dp_t0 = P[:, t] - P[:, t-1]
                dp_t1 = P[:, t-1] - P[:, t-2]
                dp_t2 = P[:, t-2] - P[:, t-3]
                
                # 3. Dynamic well controls (mapped to nodes)
                inj_BHP_node = np.zeros(n_cells)
                inj_H2_node = np.zeros(n_cells)
                inj_status_node = np.zeros(n_cells)
                
                prod_BHP_node = np.zeros(n_cells)
                prod_H2_node = np.zeros(n_cells)
                prod_status_node = np.zeros(n_cells)
                
                # Map control signals to correct well cell indices
                inj_BHP_node[self.inj_cells] = inj_BHP[t]
                inj_H2_node[self.inj_cells] = inj_H2[t]
                inj_status_node[self.inj_cells] = inj_status[t]
                
                prod_BHP_node[self.prod_cells] = prod_BHP[t]
                prod_H2_node[self.prod_cells] = prod_H2[t]
                prod_status_node[self.prod_cells] = prod_status[t]
                
                # Stack feature matrix for this timestep
                # Feature indices:
                # 0: Perm, 1: Poro, 2: Dist_to_inj, 3: Dist_to_prod
                # 4: P(t), 5: Sg(t), 6: Sw(t), 7: xH2(t), 8: yH2(t)
                # 9: dP(t), 10: dP(t-1), 11: dP(t-2)
                # 12: time_since_inj, 13: time_since_prod
                # 14: inj_BHP, 15: inj_H2, 16: inj_status
                # 17: prod_BHP, 18: prod_H2, 19: prod_status
                feat = np.column_stack((
                    perm, poro, self.dist_to_inj, self.dist_to_prod,
                    p_t, sg_t, sw_t, xh2_t, yh2_t,
                    dp_t0, dp_t1, dp_t2,
                    time_since_inj[t] * np.ones(n_cells),
                    time_since_prod[t] * np.ones(n_cells),
                    inj_BHP_node, inj_H2_node, inj_status_node,
                    prod_BHP_node, prod_H2_node, prod_status_node
                ))
                
                # Targets: P(t+1) - P(t) and Sg(t+1) - Sg(t)
                target_dp = P[:, t+1] - P[:, t]
                target_dsg = Sg[:, t+1] - Sg[:, t]
                target = np.column_stack((target_dp, target_dsg))
                
                self.samples.append({
                    'features': feat.astype(np.float32),
                    'target': target.astype(np.float32),
                    'P_t': p_t.astype(np.float32),
                    'Sg_t': sg_t.astype(np.float32),
                    'timestep': t,
                    'case': case_path.name
                })
                
    def _fit_scaler(self):
        print("    Computing feature normalization scaling...")
        all_feats = np.concatenate([s['features'] for s in self.samples], axis=0)
        means = np.mean(all_feats, axis=0)
        stds = np.std(all_feats, axis=0)
        # Avoid division by zero
        stds[stds < 1e-8] = 1.0
        
        self.scaler = {'mean': means, 'std': stds}
        
    def get_scaled_features(self, index):
        sample = self.samples[index]
        feats = (sample['features'] - self.scaler['mean']) / self.scaler['std']
        return torch.from_numpy(feats), torch.from_numpy(sample['target']), sample
        
    def __len__(self):
        return len(self.samples)
