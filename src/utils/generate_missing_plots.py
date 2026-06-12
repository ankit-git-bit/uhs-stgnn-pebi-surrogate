import numpy as np
import scipy.io as sio
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    project_root = Path(__file__).resolve().parents[2]
    data_raw_dir = project_root / "data" / "raw"
    data_processed_dir = project_root / "data" / "processed"
    results_figures_dir = project_root / "results" / "figures"
    results_metrics_dir = project_root / "results" / "metrics"
    fig_dir = project_root / "paper_project" / "figures"
    
    fig_dir.mkdir(parents=True, exist_ok=True)
    results_figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Helper to save to both locations
    def save_plot(fig, filename, dpi=600):
        fig.savefig(fig_dir / filename, dpi=dpi, bbox_inches='tight')
        fig.savefig(results_figures_dir / filename, dpi=dpi, bbox_inches='tight')
        
    def save_pdf(fig, filename):
        fig.savefig(fig_dir / filename, bbox_inches='tight')
        fig.savefig(results_figures_dir / filename, bbox_inches='tight')

    print("Generating Figure 6: Dataset Overview...")
    try:
        case1 = sio.loadmat(str(data_raw_dir / "Case_0001_wj.mat"))
        P_gt  = case1['P_matrix'].astype(np.float32)
        Sg_gt = case1['Sg_matrix'].astype(np.float32)
        inj_BHP = case1['inj_BHP'].squeeze()
        prod_BHP = case1['prod_BHP'].squeeze()
        inj_status = case1['inj_status'].squeeze()
        prod_status = case1['prod_status'].squeeze()
        
        n_steps = P_gt.shape[1]
        steps = np.arange(n_steps)
        
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        # Panel 1: Pressure
        ax = axes[0]
        ax.plot(steps, np.mean(P_gt, axis=0), color='crimson', label='Mean Pressure')
        ax.fill_between(steps, np.min(P_gt, axis=0), np.max(P_gt, axis=0), color='crimson', alpha=0.15, label='P Range [Min, Max]')
        ax.set_ylabel('Pressure (bar)', color='crimson')
        ax.tick_params(axis='y', labelcolor='crimson')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_title("UHS Reservoir Dynamic Profile: Case 1", fontsize=12, fontweight='bold')
        
        # Panel 2: Saturation
        ax = axes[1]
        ax.plot(steps, np.mean(Sg_gt, axis=0), color='navy', label='Mean Gas Saturation')
        ax.fill_between(steps, np.min(Sg_gt, axis=0), np.max(Sg_gt, axis=0), color='navy', alpha=0.15, label='Sg Range [Min, Max]')
        ax.set_ylabel('Gas Saturation (Sg)', color='navy')
        ax.tick_params(axis='y', labelcolor='navy')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Panel 3: Well controls
        ax = axes[2]
        ax.step(steps, inj_BHP, color='darkgreen', label='Inj BHP', where='mid')
        ax.step(steps, prod_BHP, color='darkorange', label='Prod BHP', where='mid')
        ax.set_ylabel('Well BHP (bar)')
        ax.set_xlabel('Simulation Timestep')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Shading operational phases based on status
        for a in axes:
            a.axvspan(0, 30, color='green', alpha=0.05) # Inj 1
            a.axvspan(30, 43, color='gray', alpha=0.05) # Shut 1
            a.axvspan(43, 73, color='orange', alpha=0.05) # Prod 1
            a.axvspan(73, 103, color='green', alpha=0.05) # Inj 2
            a.axvspan(103, 116, color='gray', alpha=0.05) # Shut 2
            a.axvspan(116, 145, color='orange', alpha=0.05) # Prod 2
            
        plt.tight_layout()
        save_plot(fig, "fig1_dataset_overview.png", dpi=600)
        save_pdf(fig, "fig1_dataset_overview.pdf")
        plt.close(fig)
        print(" -> Saved Figure 1")
    except Exception as e:
        print(f" -> Failed Figure 6: {e}")

    print("Generating Figure 7: Mesh Topology...")
    try:
        graph_data = np.load(str(data_processed_dir / "mesh_graph.npz"))
        case1 = sio.loadmat(str(data_raw_dir / "Case_0001_wj.mat"))
        X = case1['X_coords'].squeeze()
        Z = case1['Z_coords'].squeeze()
        inj_cells = graph_data['inj_cells']
        prod_cells = graph_data['prod_cells']
        
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.scatter(X, Z, s=2, color='darkgray', alpha=0.5, label='Voronoi Mesh Nodes')
        ax.scatter(X[inj_cells], Z[inj_cells], s=25, color='red', marker='^', label='Injector Well Nodes')
        ax.scatter(X[prod_cells], Z[prod_cells], s=25, color='blue', marker='v', label='Producer Well Nodes')
        ax.set_xlabel('X Coordinate (m)')
        ax.set_ylabel('Z Coordinate (m)')
        ax.set_title('Native Irregular PEBI/Voronoi Mesh Centroids (10,116 Cells)', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.2)
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        save_plot(fig, "fig2_mesh_topology.png", dpi=600)
        save_pdf(fig, "fig2_mesh_topology.pdf")
        plt.close(fig)
        print(" -> Saved Figure 2")
    except Exception as e:
        print(f" -> Failed Figure 7: {e}")

    print("Generating Figure 8: Interpolation Error Study...")
    try:
        resolutions = ['64x32', '128x64', '256x128']
        rmse_t0   = [0.65, 0.63, 0.46]
        rmse_t73  = [11.39, 11.43, 8.59]
        rmse_t145 = [8.29, 8.37, 6.30]
        
        x = np.arange(len(resolutions))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x - width, rmse_t0, width, label='t = 0 days (Initial)', color='lightblue', edgecolor='black')
        ax.bar(x, rmse_t73, width, label='t = 365 days (Peak Inject)', color='salmon', edgecolor='black')
        ax.bar(x + width, rmse_t145, width, label='t = 725 days (Peak Prod)', color='lightgreen', edgecolor='black')
        
        ax.set_xlabel('Interpolation Grid Resolution')
        ax.set_ylabel('Reconstruction RMSE (bar)')
        ax.set_title('Grid Projection Truncation Error Analysis', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(resolutions)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_plot(fig, "fig6_interpolation_error.png", dpi=600)
        save_pdf(fig, "fig6_interpolation_error.pdf")
        plt.close(fig)
        print(" -> Saved Figure 6 (reconstruction error)")
    except Exception as e:
        print(f" -> Failed Figure 8: {e}")

    print("Generating Figure 9: Baseline Comparison...")
    try:
        labels = ['Persistence', 'Linear Reg.', 'MLP (Local)', 'MLP (Spatiotemp)', 'Random Forest']
        rmse_direct = [4.218, 3.527, 3.423, np.nan, np.nan]
        rmse_change = [4.218, 4.176, 3.361, 0.906, 0.549]
        
        x = np.arange(len(labels))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - width/2, rmse_direct, width, label='Direct Pressure Target $P(t+1)$', color='steelblue', alpha=0.8, edgecolor='black')
        ax.bar(x + width/2, rmse_change, width, label='Pressure-Change Target $\\Delta P(t)$', color='darkorange', alpha=0.8, edgecolor='black')
        
        ax.set_xlabel('Model Architecture / Feature Set')
        ax.set_ylabel('Pressure Prediction RMSE (bar)')
        ax.set_title('Emulation Target Comparison: Direct vs. Pressure-Change', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_plot(fig, "fig4_baseline_comparison.png", dpi=600)
        save_pdf(fig, "fig4_baseline_comparison.pdf")
        plt.close(fig)
        print(" -> Saved Figure 4")
    except Exception as e:
        print(f" -> Failed Figure 9: {e}")

    print("Generating Figure 10: Feature Importance...")
    try:
        with open(results_metrics_dir / 'PHASE_4_FEATURE_ENGINEERING_SUMMARY.json') as f:
            fe_summary = json.load(f)
        rf_imp = fe_summary['feature_importance_rankings']['spatiotemporal']
        
        top_n = 10
        sorted_imp = sorted(rf_imp, key=lambda x: x['importance'], reverse=True)[:top_n]
        sorted_imp.reverse()
        
        names = [x['feature'] for x in sorted_imp]
        importances = [x['importance'] for x in sorted_imp]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.barh(names, importances, color='teal', edgecolor='black', alpha=0.8)
        ax.set_xlabel('Gini Feature Importance (Fraction)')
        ax.set_title('Random Forest Feature Importance: Spatiotemporal Stack', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                    f'{width*100:.1f}%',
                    ha='left', va='center', fontsize=9, fontweight='bold')
            
        ax.set_xlim(0, 0.75)
        plt.tight_layout()
        save_plot(fig, "fig5_feature_importance.png", dpi=600)
        save_pdf(fig, "fig5_feature_importance.pdf")
        plt.close(fig)
        print(" -> Saved Figure 5")
    except Exception as e:
        print(f" -> Failed Figure 10: {e}")

    print("Generating Figure 11: Pressure Evolution...")
    try:
        case1 = sio.loadmat(str(data_raw_dir / "Case_0001_wj.mat"))
        X = case1['X_coords'].squeeze()
        Z = case1['Z_coords'].squeeze()
        P = case1['P_matrix'].astype(np.float32)
        
        graph_data = np.load(str(data_processed_dir / "mesh_graph.npz"))
        inj_cells = graph_data['inj_cells']
        prod_cells = graph_data['prod_cells']
        
        timesteps = [0, 30, 43, 73, 103, 145]
        labels = [
            't = 0d (Initial)',
            't = 150d (Peak Inj 1)',
            't = 215d (Shut-in 1)',
            't = 365d (Peak Prod 1)',
            't = 515d (Peak Inj 2)',
            't = 725d (Final State)'
        ]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True, sharey=True)
        axes_flat = axes.flatten()
        vmin = P.min()
        vmax = P.max()
        
        for idx, (t, lbl) in enumerate(zip(timesteps, labels)):
            ax = axes_flat[idx]
            sc = ax.scatter(X, Z, c=P[:, t], cmap='jet', s=3, vmin=vmin, vmax=vmax, zorder=1)
            
            # Overlay well markers with high visibility (white fill, black borders)
            ax.scatter(X[inj_cells], Z[inj_cells], s=55, facecolor='white', edgecolor='black', 
                       marker='^', label='Injector' if idx == 0 else "", lw=1.2, zorder=5)
            ax.scatter(X[prod_cells], Z[prod_cells], s=55, facecolor='white', edgecolor='black', 
                       marker='v', label='Producer' if idx == 0 else "", lw=1.2, zorder=5)
            
            # Subplot titles and labels
            ax.set_title(lbl, fontsize=11, fontweight='bold', pad=8)
            ax.grid(True, alpha=0.15)
            ax.set_aspect('equal', adjustable='box')
            
            # Display legends inside subplots where appropriate
            if idx == 0:
                ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
                
            # X and Y axis labels for boundary subplots
            if idx >= 3:
                ax.set_xlabel('X Coordinate (m)', fontsize=10)
            if idx % 3 == 0:
                ax.set_ylabel('Z Coordinate (m)', fontsize=10)
            
        fig.subplots_adjust(right=0.88, hspace=0.25, wspace=0.15)
        cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(sc, cax=cbar_ax)
        cbar.set_label('Pressure (bar)', fontsize=12, fontweight='bold')
        cbar.ax.tick_params(labelsize=10)
        
        fig.suptitle('Spatial Pressure Evolution snap-shots (Case 1)', fontsize=14, fontweight='bold', y=0.96)
        
        save_plot(fig, "fig3_pressure_evolution.png", dpi=600)
        save_pdf(fig, "fig3_pressure_evolution.pdf")
        plt.close(fig)
        print(" -> Saved Figure 3")
    except Exception as e:
        print(f" -> Failed Figure 11: {e}")

    print("\n -> Missing figures generation complete!")

if __name__ == "__main__":
    main()
