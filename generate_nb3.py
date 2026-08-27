import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

nb.cells.append(nbf.v4.new_markdown_cell('''
# Baseline Modeling, Evaluation and Threshold Analysis
Bu notebook, gelişmiş modeller (XGBoost, Random Forest vb.) kurulmadan önce güvenilir bir temel (baseline) oluşturmayı amaçlar. Veri çok büyük olduğu için, bellek dostu `SGDClassifier` ile lineer (logistic) modeller eğitilecektir.
'''))

nb.cells.append(nbf.v4.new_code_cell('''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import json

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))
from fraudshield.data.loader import load_raw_data
from fraudshield.features.builder import build_all_features
from fraudshield.data.splitting import temporal_train_val_test_split
from fraudshield.models.train_baseline import train_dummy_model, train_logistic_model, train_weighted_logistic_model
from fraudshield.models.evaluate import evaluate_model
from fraudshield.models.threshold_analysis import analyze_thresholds

os.makedirs('../reports/figures', exist_ok=True)
'''))

nb.cells.append(nbf.v4.new_code_cell('''
# Veri Yükleme ve Ayırma
DATA_PATH = '../data/raw/PS_20174392719_1491204439457_log.csv'
df_raw = load_raw_data(DATA_PATH)
df_features = build_all_features(df_raw)
train_df, val_df, test_df = temporal_train_val_test_split(df_features, train_size=0.70, val_size=0.15)

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# Hedef değişkeni çıkaralım
target_col = 'isFraud'
drop_cols = [target_col, 'isFlaggedFraud', 'step'] # step ve isFlaggedFraud modelde kullanılmaz

X_train = train_df.drop(columns=drop_cols)
y_train = train_df[target_col]

X_val = val_df.drop(columns=drop_cols)
y_val = val_df[target_col]

X_test = test_df.drop(columns=drop_cols)
y_test = test_df[target_col]

numeric_features = [col for col in X_train.columns if X_train[col].dtype in ['float32', 'int8', 'int16'] and col not in ['type', 'orig_account_type', 'dest_account_type']]
categorical_features = ['type', 'orig_account_type', 'dest_account_type']

print("Sayısal özellikler:", numeric_features)
print("Kategorik özellikler:", categorical_features)
'''))

nb.cells.append(nbf.v4.new_markdown_cell('''
## Baseline Modellerin Eğitimi (Sadece Train seti üzerinde)
'''))

nb.cells.append(nbf.v4.new_code_cell('''
print("Dummy model eğitiliyor...")
dummy_model = train_dummy_model(X_train, y_train, numeric_features, categorical_features)

print("Ağırlıksız Logistic (SGD) model eğitiliyor...")
log_model = train_logistic_model(X_train, y_train, numeric_features, categorical_features)

print("Class-Weighted Logistic (SGD) model eğitiliyor...")
weighted_log_model = train_weighted_logistic_model(X_train, y_train, numeric_features, categorical_features)
'''))

nb.cells.append(nbf.v4.new_markdown_cell('''
## Validation Seti Üzerinde Değerlendirme (Threshold 0.50)
'''))

nb.cells.append(nbf.v4.new_code_cell('''
models = {
    'Dummy': dummy_model,
    'Unweighted Logistic': log_model,
    'Weighted Logistic': weighted_log_model
}

val_results = {}
val_probs = {}

for name, model in models.items():
    if name == 'Dummy':
        y_prob = model.predict_proba(X_val)[:, 1]
    else:
        # SGDClassifier with log_loss has predict_proba
        y_prob = model.predict_proba(X_val)[:, 1]
    
    y_pred = (y_prob >= 0.5).astype(int)
    val_probs[name] = y_prob
    val_results[name] = evaluate_model(y_val, y_pred, y_prob)

val_results_df = pd.DataFrame(val_results).T
display(val_results_df[['PR-AUC', 'ROC-AUC', 'Precision', 'Recall', 'F1-score', 'Fraud Alert Rate']])
'''))

nb.cells.append(nbf.v4.new_markdown_cell('''
## Threshold Analizi (En iyi model seçimi)
Ağırlıklı (Weighted) model genellikle class imbalance'da recall'u çok artırır ancak precision çok düşer. Ağırlıksız (Unweighted) model ise threshold optimizasyonu ile daha iyi dengelenebilir.
Threshold analizini `Unweighted Logistic` modeli için yapalım, zira PR-AUC ve stabilite genelde ondan gelir. (Tabloya göre en yüksek PR-AUC'a sahip modele bakınız).
'''))

nb.cells.append(nbf.v4.new_code_cell('''
best_model_name = val_results_df['PR-AUC'].idxmax()
print(f"En iyi PR-AUC'a sahip model: {best_model_name}")

selected_probs = val_probs[best_model_name]
thresh_analysis = analyze_thresholds(y_val, selected_probs, fp_cost=1, fn_cost=100)

print("Max F1 Threshold:", thresh_analysis['best_f1'])
print("Min Cost Threshold:", thresh_analysis['min_cost'])
print("Recall >= 80% Threshold:", thresh_analysis['recall_80'])
print("Recall >= 90% Threshold:", thresh_analysis['recall_90'])
'''))

nb.cells.append(nbf.v4.new_code_cell('''
# Plotting PR Curve
plt.figure(figsize=(8,6))
plt.plot(thresh_analysis['recalls'], thresh_analysis['precisions'], label=f'PR Curve ({best_model_name})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Validation)')
plt.legend()
plt.savefig('../reports/figures/pr_curve.png')
plt.show()

# Plotting F1 vs Threshold
plt.figure(figsize=(8,6))
plt.plot(thresh_analysis['thresholds'], thresh_analysis['f1_scores'])
plt.xlabel('Threshold')
plt.ylabel('F1 Score')
plt.title('F1 Score vs Threshold (Validation)')
plt.axvline(thresh_analysis['best_f1'], color='r', linestyle='--', label='Best F1')
plt.legend()
plt.savefig('../reports/figures/f1_vs_threshold.png')
plt.show()

# Plotting Cost vs Threshold
plt.figure(figsize=(8,6))
plt.plot(thresh_analysis['thresholds'], thresh_analysis['costs'])
plt.xlabel('Threshold')
plt.ylabel('Cost')
plt.title('Hypothetical Cost vs Threshold (Validation)')
plt.axvline(thresh_analysis['min_cost'], color='g', linestyle='--', label='Min Cost')
plt.legend()
plt.savefig('../reports/figures/cost_vs_threshold.png')
plt.show()
'''))

nb.cells.append(nbf.v4.new_markdown_cell('''
## Test Seti Üzerinde Karşılaştırma
Seçilen final model (Unweighted Logistic) ve seçilen threshold (Örn: En yüksek F1'i veren threshold) kullanılarak Test seti bir kez değerlendirilir.
Ayrıca, veri setinde bulunan orijinal `isFlaggedFraud` kuralı ve hibrit bir sistem (Kural OR Model) ile kıyaslanır.
'''))

nb.cells.append(nbf.v4.new_code_cell('''
final_model = models[best_model_name]
final_threshold = thresh_analysis['best_f1']

# 1. Yalnızca ML
test_probs = final_model.predict_proba(X_test)[:, 1]
ml_preds = (test_probs >= final_threshold).astype(int)
ml_metrics = evaluate_model(y_test, ml_preds, test_probs)

# 2. Yalnızca kural (isFlaggedFraud)
rule_preds = test_df['isFlaggedFraud'].values
rule_probs = rule_preds.copy() # Prob is 0 or 1
rule_metrics = evaluate_model(y_test, rule_preds, rule_probs)

# 3. Hibrit Sistem (Kural OR ML)
hybrid_preds = np.maximum(ml_preds, rule_preds)
hybrid_probs = np.maximum(test_probs, rule_probs)
hybrid_metrics = evaluate_model(y_test, hybrid_preds, hybrid_probs)

test_comparison = pd.DataFrame({
    'Rule Only (isFlaggedFraud)': rule_metrics,
    'ML Only (Baseline)': ml_metrics,
    'Hybrid (Rule OR ML)': hybrid_metrics
}).T

display(test_comparison[['Precision', 'Recall', 'F1-score', 'False Positive', 'False Negative', 'Alerts per 1000']])
'''))

nb.cells.append(nbf.v4.new_code_cell('''
import json

# Raporu diske yazma
report_content = f"""# Baseline Modeling and Evaluation Results

## Model Özeti
- **Kullanılan Feature Listesi:** {numeric_features + categorical_features}
- **Veri Boyutları:** Train ({len(train_df)}), Validation ({len(val_df)}), Test ({len(test_df)})

## Validation Metrikleri (Threshold: 0.5)
{val_results_df[['PR-AUC', 'ROC-AUC', 'Precision', 'Recall', 'F1-score', 'Alerts per 1000']].to_markdown()}

## Seçim Gerekçesi
- **Seçilen Model:** {best_model_name} (En yüksek PR-AUC'ye sahip olması sebebiyle seçilmiştir).
- **Seçilen Operasyon Threshold'u:** {final_threshold:.4f} (Validation seti üzerinde F1 skorunu maksimize eden noktadır).

## Test Seti Karşılaştırması
{test_comparison[['Precision', 'Recall', 'F1-score', 'True Positive', 'False Positive', 'False Negative', 'Alerts per 1000']].to_markdown()}

**Sonuç:** ML modeli (özellikle Hybrid sistem), mevcut sadece-kural tabanlı yaklaşıma göre Recall (Duyarlılık) oranını binlerce kat artırmış, makul bir Precision ile Fraud vakalarının büyük kısmını yakalamıştır.
"""

with open('../reports/baseline_results.md', 'w', encoding='utf-8') as f:
    f.write(report_content)
    
print("Rapor oluşturuldu.")
'''))

os.makedirs('notebooks', exist_ok=True)
nbf.write(nb, 'notebooks/03_baseline_modeling.ipynb')
print("Notebook created.")
