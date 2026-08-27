import os
import json

notebooks = [
    'notebooks/01_data_validation_and_eda.ipynb',
    'notebooks/02_feature_engineering_and_split.ipynb',
    'notebooks/03_baseline_modeling.ipynb'
]

replacements = {
    "src.data.load_data": "fraudshield.data.loader",
    "src.data.validate_data": "fraudshield.data.validation",
    "src.data.split_data": "fraudshield.data.splitting",
    "src.features.build_features": "fraudshield.features.builder",
    "src.models.train_baseline": "fraudshield.models.train_baseline",
    "src.models.evaluate": "fraudshield.models.evaluate",
    "src.models.threshold_analysis": "fraudshield.models.threshold_analysis"
}

for nb_path in notebooks:
    if not os.path.exists(nb_path): continue
    
    with open(nb_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    for old, new in replacements.items():
        content = content.replace(old, new)
        
    with open(nb_path, 'w', encoding='utf-8') as f:
        f.write(content)

print("Notebooks updated.")
