import os
import json

runs_dir = 'artifacts/runs'
for run in os.listdir(runs_dir):
    p = os.path.join(runs_dir, run, 'best_params.json')
    if os.path.exists(p):
        with open(p) as f:
            d = json.load(f)
            print(f"Run: {run}")
            print(f"  Model: {d.get('model_type', 'MLP')}")
            print(f"  Loss: {d.get('loss_type', 'N/A')}")
            print(f"  Feature Set in config?: {d.get('feature_set_name', 'N/A')}")
    
    p2 = os.path.join(runs_dir, run, 'architecture.json')
    if os.path.exists(p2):
        with open(p2) as f:
            d = json.load(f)
            print(f"  Arch input_dim: {d.get('input_dim')}")
            print(f"  Feature set from arch: {d.get('feature_set_name', 'N/A')}")

    p3 = os.path.join(runs_dir, run, 'training_history.json')
    if os.path.exists(p3):
        with open(p3) as f:
            d = json.load(f)
            print(f"  Metrics: {list(d.get('val_metrics', {}).keys())}")
            if 'val_metrics' in d:
                print(f"  PR-AUC: {d['val_metrics'].get('pr_auc')}")
