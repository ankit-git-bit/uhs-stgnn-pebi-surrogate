import os

def reconstruct_final_paper():
    preamble = r"""\documentclass[10pt,twocolumn]{article}
\usepackage[utf8]{inputenc}
\usepackage{geometry}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{graphicx}
% \usepackage{cite}           % Commented out to resolve package conflict with natbib in Overleaf
\usepackage{authblk}

% --- Essential Packages Added for Typesetting and Layout Stability ---
\usepackage{microtype}        % Improves character spacing and reduces overfull box warnings
\usepackage{bm}               % Bold math support (vectors/tensors)
\usepackage{xcolor}           % Colors for highlights and links
\usepackage{float}            % Tight floating control [H]
\usepackage{tabularx}         % Self-adjusting table column widths
\usepackage{caption}          % Caption styling and control
\usepackage{url}              % Clean URL breaks
\usepackage{subcaption}       % Subfigure support for multi-panel layouts

\geometry{margin=0.75in, top=0.85in, bottom=0.85in}

\title{\textbf{Spatio-Temporal Graph Neural Networks for Underground Hydrogen Storage Forecasting on Irregular PEBI Meshes}}

\author{\textbf{Anonymous Authors}}
\affil{Anonymous Institution}
\date{}

% --- Template for Final Camera-Ready Author Block ---
% \author[1]{\textbf{KTU Internship Project Team}}
% \affil[1]{Department of Scientific Computing \& Machine Learning}
% \date{}

\begin{document}
\maketitle
"""

    sections = [
        'abstract.tex',
        'introduction.tex',
        'physics.tex',
        'dataset.tex',
        'baseline_models.tex',
        'pressure_change.tex',
        'feature_engineering.tex',
        'graph_surrogate.tex',
        'stgnn.tex',
        'rollout_results.tex',
        'discussion.tex',
        'practical_implications.tex',
        'limitations.tex',
        'conclusions.tex'
    ]
    
    supplementary = [
        'nomenclature.tex',
        'hyperparameters.tex',
        'reproducibility.tex'
    ]
    
    content_parts = [preamble]
    
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Read sections
    for sec in sections:
        path = os.path.join(project_root, 'paper_project', 'sections', sec)
        with open(path, 'r', encoding='utf-8') as f:
            sec_content = f.read().strip()
        # Add comment separators
        content_parts.append(f"\n% ==========================================\n% SECTION: {sec.replace('.tex', '').upper()}\n% ==========================================\n")
        content_parts.append(sec_content)
        content_parts.append("\n")
        
    # Read appendix/supplementary
    content_parts.append("\n\\clearpage\n\\appendix\n")
    for supp in supplementary:
        path = os.path.join(project_root, 'paper_project', 'supplementary', supp)
        with open(path, 'r', encoding='utf-8') as f:
            supp_content = f.read().strip()
        content_parts.append(f"\n% ==========================================\n% SUPPLEMENTARY: {supp.replace('.tex', '').upper()}\n% ==========================================\n")
        content_parts.append(supp_content)
        content_parts.append("\n")
        
    # Read bibliography
    footer = r"""
\small
\bibliographystyle{ieeetr}
\bibliography{references}

\end{document}
"""
    content_parts.append(footer)
    
    full_text = "".join(content_parts)
    
    # Write to final_paper.tex at project root
    output_path = os.path.join(project_root, 'final_paper.tex')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_text)
    print(f"Successfully reconstructed {output_path}!")

if __name__ == "__main__":
    reconstruct_final_paper()
