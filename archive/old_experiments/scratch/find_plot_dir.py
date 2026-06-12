with open('evaluate_rollout_v3.py', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if 'plot_dir' in line:
            print(f"Line {i+1}: {line.strip()}")
