import os
import shutil
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def main():
    dest_dir = "paper_figure"
    os.makedirs(dest_dir, exist_ok=True)
    print(f"Created directory: {dest_dir}")

    # 1. Copy existing figures with correct target names
    mappings = [
        ("dataset_overview.png", "dataset_overview.png"),
        ("paper_project/figures/fig11_pressure_evolution.png", "pressure_snapshots.png"),
        ("phase4_plots/model_performance_comparison.png", "model_performance_comparison.png"),
        ("paper_project/figures/fig10_feature_importance.png", "feature_importance.png"),
        ("paper_project/figures/fig7_mesh_topology.png", "mesh_topology.png"),
        ("paper_project/figures/fig2_rollout_rmse.png", "rollout_rmse.png"),
        ("paper_project/figures/fig3_pressure_tracking.png", "pressure_track.png"),
        ("phase6_plots/v1_vs_v2_rollout_rmse.png", "rollout_growth.png")
    ]

    for src, dest in mappings:
        if os.path.exists(src):
            shutil.copy(src, os.path.join(dest_dir, dest))
            print(f"Successfully copied: {src} -> {dest_dir}/{dest}")
        else:
            print(f"Warning: Source file {src} not found!")

    # 2. Generate sweep_results.png using actual sweep data
    try:
        print("Generating sweep_results.png...")
        configs = ["0.5P+0.5Sg", "0.7P+0.3Sg", "0.3P+0.7Sg"]
        mean_P_rmse = [27.086, 22.291, 37.261]
        mean_Sg_rmse = [0.02327, 0.02407, 0.03244]
        balance_scores = [0.00630, 0.00536, 0.01209]

        fig, ax1 = plt.subplots(figsize=(8, 5))

        color = 'steelblue'
        ax1.set_xlabel('Loss Weight Configuration (alpha_P * L_P + alpha_Sg * L_Sg)')
        ax1.set_ylabel('Mean Pressure RMSE (bar)', color=color)
        bars = ax1.bar(configs, mean_P_rmse, color=color, width=0.4, alpha=0.8, edgecolor='black', label='Pressure RMSE')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, alpha=0.3, axis='y')

        # Instantiate a second axes that shares the same x-axis
        ax2 = ax1.twinx()  
        color = 'darkorange'
        ax2.set_ylabel('Mean Gas Saturation RMSE (Sg)', color=color)
        ax2.plot(configs, mean_Sg_rmse, color=color, marker='o', linewidth=2, markersize=8, label='Saturation RMSE')
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title('Loss-Weight Hyperparameter Sweep Performance (ST-GNN v3)', fontsize=12, fontweight='bold')
        fig.tight_layout()
        plt.savefig(os.path.join(dest_dir, "sweep_results.png"), dpi=600)
        plt.close(fig)
        print(" -> Saved sweep_results.png")
    except Exception as e:
        print(f" -> Failed sweep_results.png: {e}")

    # 3. Generate stgnn_architecture.png as a professional block diagram schematic
    try:
        print("Generating stgnn_architecture.png...")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.axis('off')

        # Draw blocks
        # 1. Inputs Block
        ax.add_patch(patches.FancyBboxPatch((0.5, 2.0), 1.5, 2.0, boxstyle="round,pad=0.1", facecolor="lightgray", edgecolor="black"))
        ax.text(1.25, 3.5, "Node Features\n(Static Rock,\nCoordinates,\nControls, Lags)\n[h_i(t)]", ha="center", va="center", fontsize=9, fontweight="bold")
        ax.text(1.25, 2.2, "Edge Features\n(T_ij, dist_ij)\n[e_ij]", ha="center", va="center", fontsize=8, color="dimgray")

        # Arrow 1
        ax.annotate("", xy=(2.7, 3.0), xytext=(2.2, 3.0), arrowprops=dict(arrowstyle="->", lw=1.5))

        # 2. Transmissibility Aggregator
        ax.add_patch(patches.FancyBboxPatch((2.8, 1.8), 2.0, 2.4, boxstyle="round,pad=0.1", facecolor="#e6f2ff", edgecolor="#0066cc"))
        ax.text(3.8, 3.6, "Physical Message Passing", ha="center", va="center", fontsize=9, fontweight="bold", color="#004080")
        ax.text(3.8, 2.8, "T_ij Weighted\nGraph Convolution\n(Eq. 7)", ha="center", va="center", fontsize=8)
        ax.text(3.8, 2.0, "LeakyReLU & LayerNorm\n+ Residual Paths", ha="center", va="center", fontsize=8, color="dimgray")

        # Arrow 2
        ax.annotate("", xy=(5.5, 3.0), xytext=(4.9, 3.0), arrowprops=dict(arrowstyle="->", lw=1.5))

        # 3. Temporal Updates GRU
        ax.add_patch(patches.FancyBboxPatch((5.6, 2.0), 1.6, 2.0, boxstyle="round,pad=0.1", facecolor="#ffe6e6", edgecolor="#cc0000"))
        ax.text(6.4, 3.0, "Temporal GRU\nCell\n(Tracks hidden\ntransient state\nh_GRU)", ha="center", va="center", fontsize=9, fontweight="bold", color="#800000")

        # Arrow 3
        ax.annotate("", xy=(7.9, 3.0), xytext=(7.3, 3.0), arrowprops=dict(arrowstyle="->", lw=1.5))

        # 4. Decoder / Output Head
        ax.add_patch(patches.FancyBboxPatch((8.0, 2.2), 1.6, 1.6, boxstyle="round,pad=0.1", facecolor="#e6ffe6", edgecolor="#00cc00"))
        ax.text(8.8, 3.0, "Dense Decoder\n[MLP_dec]\nPredictions:\n[dP_i, dS_g,i]", ha="center", va="center", fontsize=9, fontweight="bold", color="#006600")

        plt.title('Spatio-Temporal Graph Neural Network (ST-GNN) Architecture', fontsize=13, fontweight='bold', y=0.95)
        plt.tight_layout()
        plt.savefig(os.path.join(dest_dir, "stgnn_architecture.png"), dpi=600, bbox_inches='tight')
        plt.savefig(os.path.join(dest_dir, "stgnn_architecture.pdf"), bbox_inches='tight')
        plt.close(fig)
        print(" -> Saved stgnn_architecture.png")
    except Exception as e:
        print(f" -> Failed stgnn_architecture.png: {e}")

    print("\n[DONE] Paper figures created/copied in paper_figure folder!")

if __name__ == "__main__":
    main()
