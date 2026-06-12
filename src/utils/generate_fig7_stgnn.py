import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def generate_fig7():
    project_root = Path(__file__).resolve().parents[2]
    fig_dir = project_root / "paper_project" / "figures"
    results_figures_dir = project_root / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    results_figures_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.set_xlim(0, 14.5)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # helper to add text with shadow/outline or clean fonts
    def draw_box(x, y, w, h, text, fc, ec, title_color="black", text_size=9, title_size=10):
        # Background box
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", 
                                     facecolor=fc, edgecolor=ec, lw=1.5, zorder=2)
        ax.add_patch(box)
        # Text inside
        lines = text.split('\n')
        title = lines[0]
        body = '\n'.join(lines[1:])
        
        ax.text(x + w/2, y + h - 0.35, title, ha="center", va="center", 
                fontsize=title_size, fontweight="bold", color=title_color, zorder=3)
        if body:
            ax.text(x + w/2, y + h/2 - 0.2, body, ha="center", va="center", 
                    fontsize=text_size, color="black", zorder=3)
            
    # Draw blocks
    # 1. Inputs Block
    draw_box(0.3, 1.8, 1.3, 2.4, 
             "Node Features\n(20 Channels)\n\n• Rock Perm/Poro\n• BHP / Wells\n• Temp Lags\n• Well Times", 
             "#f2f2f2", "#595959", title_color="#333333", text_size=8, title_size=9)
    
    # 2. Graph Conv Blocks
    # Darcy GCN 1
    draw_box(2.2, 1.8, 1.3, 2.4, 
             "GCN Layer 1\n(128 Channels)\n\nDarcyConv\n+ LeakyReLU\n+ LayerNorm", 
             "#e6f2ff", "#0066cc", title_color="#004080", text_size=8, title_size=9)
    # Darcy GCN 2
    draw_box(4.1, 1.8, 1.3, 2.4, 
             "GCN Layer 2\n(128 Channels)\n\nDarcyConv\n+ LeakyReLU\n+ LayerNorm", 
             "#e6f2ff", "#0066cc", title_color="#004080", text_size=8, title_size=9)
    # Darcy GCN 3
    draw_box(6.0, 1.8, 1.3, 2.4, 
             "GCN Layer 3\n(128 Channels)\n\nDarcyConv\n+ LeakyReLU\n+ LayerNorm", 
             "#e6f2ff", "#0066cc", title_color="#004080", text_size=8, title_size=9)
    # Darcy GCN 4
    draw_box(7.9, 1.8, 1.3, 2.4, 
             "GCN Layer 4\n(128 Channels)\n\nDarcyConv\n+ LeakyReLU\n+ LayerNorm", 
             "#e6f2ff", "#0066cc", title_color="#004080", text_size=8, title_size=9)
    
    # 3. GRU Block
    draw_box(9.8, 1.8, 1.2, 2.4, 
             "Temporal GRU\n(128 Channels)\n\nTracks recurrent\ntransient state\nover timesteps", 
             "#ffe6e6", "#cc0000", title_color="#800000", text_size=8, title_size=9)
    
    # 4. Decoder Block
    draw_box(11.6, 1.8, 1.2, 2.4, 
             "MLP Decoder\n(128 → 64 → 2)\n\n3 Dense Layers\n+ LeakyReLU\n+ Dropout", 
             "#e6ffe6", "#00cc00", title_color="#006600", text_size=8, title_size=9)
    
    # 5. Output Node
    draw_box(13.4, 2.2, 0.8, 1.6, 
             "Outputs\n\nΔP_i(t)\nΔS_g,i(t)", 
             "#ffffff", "#333333", title_color="#1a1a1a", text_size=9, title_size=9)

    # Draw arrows and labels
    arrow_style = dict(arrowstyle="->", color="#333333", lw=1.5)
    
    # Main flow arrows
    ax.annotate("", xy=(2.2, 3.0), xytext=(1.6, 3.0), arrowprops=arrow_style)
    ax.annotate("", xy=(4.1, 3.0), xytext=(3.5, 3.0), arrowprops=arrow_style)
    ax.annotate("", xy=(6.0, 3.0), xytext=(5.4, 3.0), arrowprops=arrow_style)
    ax.annotate("", xy=(7.9, 3.0), xytext=(7.3, 3.0), arrowprops=arrow_style)
    ax.annotate("", xy=(9.8, 3.0), xytext=(9.2, 3.0), arrowprops=arrow_style)
    ax.annotate("", xy=(11.6, 3.0), xytext=(11.0, 3.0), arrowprops=arrow_style)
    ax.annotate("", xy=(13.4, 3.0), xytext=(12.8, 3.0), arrowprops=arrow_style)
    
    # Physical weight inputs (Transmissibility)
    trans_box = patches.FancyBboxPatch((4.0, 0.3), 1.5, 0.8, boxstyle="round,pad=0.05",
                                         facecolor="#fff7e6", edgecolor="#ff9900", lw=1.2, zorder=2)
    ax.add_patch(trans_box)
    ax.text(4.75, 0.7, "Transmissibility Matrix\nEdge Weights (E) [T_ij]", 
            ha="center", va="center", fontsize=8, color="#804d00", fontweight="bold")
    
    # Arrows from Transmissibility to each GCN layer
    gcn_centers = [2.85, 4.75, 6.65, 8.55]
    for gc in gcn_centers:
        ax.annotate("", xy=(gc, 1.8), xytext=(4.75, 1.1),
                    arrowprops=dict(arrowstyle="->", color="#ff9900", lw=1.0, ls=":"), zorder=1)
        
    # Residual connections (curved arrows)
    def draw_residual(start_x, end_x, y_level=4.3, rad=0.5):
        arrow = patches.FancyArrowPatch((start_x, y_level), (end_x, y_level),
                                         connectionstyle=f"arc3,rad={rad}",
                                         arrowstyle="-|>", mutation_scale=12,
                                         color="#0066cc", lw=1.2, ls="--", zorder=3)
        ax.add_patch(arrow)
        # plus sign for residual addition
        ax.plot(end_x, y_level, 'o', color="#0066cc", markersize=4, zorder=4)
        ax.text((start_x + end_x)/2, y_level + 0.35, "+ Residual", 
                ha="center", va="center", fontsize=7.5, color="#004080", fontweight="bold")
        
    # Add residual connections over GCN 1 to GCN 4
    draw_residual(2.0, 3.65, rad=0.35)
    draw_residual(3.9, 5.55, rad=0.35)
    draw_residual(5.8, 7.45, rad=0.35)
    draw_residual(7.7, 9.35, rad=0.35)
    
    # Annotations/Labels
    ax.text(2.85, 4.9, "128", ha="center", va="center", fontsize=9, color="#0066cc", fontweight="bold")
    ax.text(4.75, 4.9, "128", ha="center", va="center", fontsize=9, color="#0066cc", fontweight="bold")
    ax.text(6.65, 4.9, "128", ha="center", va="center", fontsize=9, color="#0066cc", fontweight="bold")
    ax.text(8.55, 4.9, "128", ha="center", va="center", fontsize=9, color="#0066cc", fontweight="bold")
    ax.text(1.9, 3.2, "20", ha="center", va="center", fontsize=9, color="black", fontweight="bold")
    ax.text(10.4, 4.9, "128", ha="center", va="center", fontsize=9, color="#cc0000", fontweight="bold")
    ax.text(12.2, 4.9, "128", ha="center", va="center", fontsize=9, color="#006600", fontweight="bold")
    
    plt.title('Spatio-Temporal Graph Neural Network (ST-GNN) Architecture Schematic', 
              fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    
    fig.savefig(fig_dir / "fig7_stgnn_architecture.png", dpi=600, bbox_inches='tight')
    fig.savefig(results_figures_dir / "fig7_stgnn_architecture.png", dpi=600, bbox_inches='tight')
    fig.savefig(fig_dir / "fig7_stgnn_architecture.pdf", bbox_inches='tight')
    fig.savefig(results_figures_dir / "fig7_stgnn_architecture.pdf", bbox_inches='tight')
    plt.close(fig)
    print(" -> Saved Figure 7 (stgnn architecture) to results/figures/ and paper_project/figures/")

if __name__ == "__main__":
    generate_fig7()
