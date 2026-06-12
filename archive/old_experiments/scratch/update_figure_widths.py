import os
import re

sections_dir = 'paper_project/sections'

adjustments = {
    'dataset.tex': [
        (r'\\begin\{figure\*\}\[t\]\s*\\centering\s*\\includegraphics\[width=[^\]]+\]\{figures/fig1_dataset_overview\}', 
         '\\begin{figure*}[t]\n    \\centering\n    \\includegraphics[width=\\textwidth]{figures/fig1_dataset_overview}'),
        (r'\\begin\{figure\}\[htbp\]\s*\\centering\s*\\includegraphics\[width=[^\]]+\]\{figures/fig2_mesh_topology\}', 
         '\\begin{figure}[htbp]\n    \\centering\n    \\includegraphics[width=0.95\\columnwidth]{figures/fig2_mesh_topology}'),
        (r'\\begin\{figure\*\}\[t\]\s*\\centering\s*\\includegraphics\[width=[^\]]+\]\{figures/fig3_pressure_evolution\}', 
         '\\begin{figure*}[t]\n    \\centering\n    \\includegraphics[width=\\textwidth]{figures/fig3_pressure_evolution}'),
    ],
    'baseline_models.tex': [
        (r'\\begin\{figure\}\[htbp\]\s*\\centering\s*\\includegraphics\[width=[^\]]+\]\{figures/fig4_baseline_comparison\}', 
         '\\begin{figure}[htbp]\n    \\centering\n    \\includegraphics[width=0.95\\columnwidth]{figures/fig4_baseline_comparison}'),
    ],
    'feature_engineering.tex': [
        (r'\\begin\{figure\}\[htbp\]\s*\\centering\s*\\includegraphics\[width=[^\]]+\]\{figures/fig5_feature_importance\}', 
         '\\begin{figure}[htbp]\n    \\centering\n    \\includegraphics[width=0.95\\columnwidth]{figures/fig5_feature_importance}'),
    ],
    'graph_surrogate.tex': [
        (r'\\begin\{figure\}\[htbp\]\s*\\centering\s*\\includegraphics\[width=[^\]]+\]\{figures/fig6_interpolation_error\}', 
         '\\begin{figure}[htbp]\n    \\centering\n    \\includegraphics[width=0.95\\columnwidth]{figures/fig6_interpolation_error}'),
    ],
    'stgnn.tex': [
        (r'\\begin\{figure\}\[htbp\]\s*\\centering\s*\\includegraphics\[width=[^\]]+\]\{figures/fig7_stgnn_architecture\}', 
         '\\begin{figure}[htbp]\n    \\centering\n    \\includegraphics[width=0.95\\columnwidth]{figures/fig7_stgnn_architecture}'),
    ],
    'rollout_results.tex': [
        (r'\\begin\{figure\}\[htbp\]\s*\\centering\s*\\includegraphics\[width=[^\]]+\]\{figures/fig8_training_curves\}', 
         '\\begin{figure}[htbp]\n    \\centering\n    \\includegraphics[width=0.95\\columnwidth]{figures/fig8_training_curves}'),
        (r'\\begin\{figure\}\[htbp\]\s*\\centering\s*\\includegraphics\[width=[^\]]+\]\{figures/fig9_rollout_rmse\}', 
         '\\begin{figure}[htbp]\n    \\centering\n    \\includegraphics[width=0.95\\columnwidth]{figures/fig9_rollout_rmse}'),
        (r'\\begin\{figure\*\}\[t\]\s*\\centering\s*\\includegraphics\[width=[^\]]+\]\{figures/fig10_pressure_tracking\}', 
         '\\begin{figure*}[t]\n    \\centering\n    \\includegraphics[width=\\textwidth]{figures/fig10_pressure_tracking}'),
        (r'\\begin\{figure\*\}\[t\]\s*\\centering\s*\\includegraphics\[width=[^\]]+\]\{figures/fig11_saturation_tracking\}', 
         '\\begin{figure*}[t]\n    \\centering\n    \\includegraphics[width=\\textwidth]{figures/fig11_saturation_tracking}'),
        # fig12 (change from figure to figure*)
        (r'\\begin\{figure\}\[htbp\]\s*\\centering\s*\\includegraphics\[width=[^\]]+\]\{figures/fig12_bar_comparison\}(.*?)\\end\{figure\}', 
         '\\begin{figure*}[t]\n    \\centering\n    \\includegraphics[width=\\textwidth]{figures/fig12_bar_comparison}\\1\\end{figure*}'),
    ]
}

for file, file_adjusts in adjustments.items():
    path = os.path.join(sections_dir, file)
    if not os.path.exists(path):
        print(f"Error: {path} not found")
        continue
        
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    original = content
    for pattern, replacement in file_adjusts:
        rx = re.compile(pattern, re.DOTALL)
        
        # Define a lambda that handles literal replacements and group back-references safely
        def make_repl(m, repl=replacement):
            if '\\1' in repl:
                return repl.replace('\\1', m.group(1))
            return repl
            
        content, count = rx.subn(make_repl, content)
        if count > 0:
            print(f"  Applied adjustment to {file} ({count} matches replaced)")
        else:
            print(f"  Warning: No matches for adjustment pattern in {file}")
            
    if content != original:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Successfully saved updates to {path}")
