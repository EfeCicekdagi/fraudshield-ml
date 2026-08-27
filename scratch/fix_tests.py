import os
import shutil

tests = [
    'test_data_split.py',
    'test_data_validation.py',
    'test_evaluation.py',
    'test_feature_engineering.py',
    'test_threshold_analysis.py'
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

os.makedirs('tests/unit', exist_ok=True)
os.makedirs('tests/integration', exist_ok=True)

for test_file in tests:
    src_path = os.path.join('tests', test_file)
    dst_path = os.path.join('tests/unit', test_file)
    
    if not os.path.exists(src_path):
        continue
        
    with open(src_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    for old, new in replacements.items():
        content = content.replace(old, new)
        
    with open(dst_path, 'w', encoding='utf-8') as f:
        f.write(content)
        
    os.remove(src_path)

print("Tests moved and updated.")
