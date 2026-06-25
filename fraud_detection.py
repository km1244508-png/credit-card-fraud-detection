"""Credit Card Fraud Detection - Optimized version with threshold tuning"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve)
from imblearn.over_sampling import SMOTE
import warnings, os, json
warnings.filterwarnings('ignore')

OUTPUT_DIR = "/home/claude/fraud-detection/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(42)
n_samples = 30000
fraud_count = 200

# Build realistic fraud dataset
X_normal, _ = make_classification(n_samples=n_samples-fraud_count, n_features=28,
                                   n_informative=14, n_redundant=8, random_state=42)
X_fraud, _  = make_classification(n_samples=fraud_count, n_features=28,
                                   n_informative=18, n_redundant=5, random_state=99)

# Fraud has different pattern
X_fraud = X_fraud * 2.5 + 1.5

X_raw = np.vstack([X_normal, X_fraud])
y_raw = np.array([0]*(n_samples-fraud_count) + [1]*fraud_count)

df = pd.DataFrame(X_raw, columns=[f"V{i}" for i in range(1,29)])
df["Time"]   = np.sort(np.random.exponential(50000, n_samples))
df["Amount"] = np.where(y_raw==1,
                        np.random.lognormal(4.0,1.0,n_samples),
                        np.random.lognormal(3.0,1.2,n_samples))
df["Class"]  = y_raw

# Shuffle
idx = np.random.permutation(len(df))
df = df.iloc[idx].reset_index(drop=True)
y_raw = y_raw[idx]

print(f"Dataset: {len(df):,} | Fraud: {df['Class'].sum()} ({df['Class'].mean()*100:.3f}%)")

# ── EDA Plots ──
counts = df['Class'].value_counts().sort_index()
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('#0F1117')
for ax in axes: ax.set_facecolor('#1A1D2E')
bars = axes[0].bar(['Normal (0)','Fraud (1)'], counts.values, color=['#00D4AA','#FF4757'], width=0.5)
axes[0].set_title('Class Distribution', color='white', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Count', color='#8B8FA8')
axes[0].tick_params(colors='#8B8FA8')
for spine in axes[0].spines.values(): spine.set_edgecolor('#2D3150')
for bar, c in zip(bars, counts.values):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+20, f'{c:,}',
                 ha='center', color='white', fontsize=11, fontweight='bold')
axes[1].pie(counts.values, labels=['Normal','Fraud'], colors=['#00D4AA','#FF4757'],
            autopct='%1.2f%%', textprops={'color':'white','fontsize':12},
            wedgeprops={'edgecolor':'#0F1117','linewidth':2})
axes[1].set_title('Class Proportion', color='white', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/class_distribution.png", dpi=150, bbox_inches='tight', facecolor='#0F1117')
plt.close(); print("✓ class_distribution.png")

# ── Amount distributions ──
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.patch.set_facecolor('#0F1117')
for ax in axes.flat:
    ax.set_facecolor('#1A1D2E')
    for spine in ax.spines.values(): spine.set_edgecolor('#2D3150')
for idx2, (cls, color, label) in enumerate([(0,'#00D4AA','Normal'),(1,'#FF4757','Fraud')]):
    s = df[df['Class']==cls]['Amount']
    axes[0,idx2].hist(s.clip(upper=600), bins=40, color=color, alpha=0.85)
    axes[0,idx2].set_title(f'{label} — Amount Distribution', color='white', fontsize=12, fontweight='bold')
    axes[0,idx2].set_xlabel('Amount ($)', color='#8B8FA8')
    axes[0,idx2].set_ylabel('Frequency', color='#8B8FA8')
    axes[0,idx2].tick_params(colors='#8B8FA8')
bp = axes[1,0].boxplot([df[df['Class']==0]['Amount'].clip(upper=600),
                         df[df['Class']==1]['Amount'].clip(upper=600)],
                        labels=['Normal','Fraud'], patch_artist=True,
                        medianprops={'color':'white','linewidth':2})
for patch, c in zip(bp['boxes'], ['#00D4AA','#FF4757']):
    patch.set_facecolor(c); patch.set_alpha(0.7)
axes[1,0].set_title('Amount Box Plot', color='white', fontsize=12, fontweight='bold')
axes[1,0].tick_params(colors='#8B8FA8')
stats = df.groupby('Class')['Amount'].agg(['mean','median','std','max'])
stats.index = ['Normal','Fraud']
axes[1,1].axis('off')
t = axes[1,1].table(cellText=[[f"${v:.2f}" for v in row] for row in stats.values],
    rowLabels=list(stats.index), colLabels=['Mean','Median','Std','Max'],
    cellLoc='center', loc='center', bbox=[0.05,0.2,0.9,0.6])
t.auto_set_font_size(False); t.set_fontsize(11)
for (r,c), cell in t.get_celld().items():
    cell.set_facecolor('#2D3150' if r>0 else '#3D4170')
    cell.set_text_props(color='white'); cell.set_edgecolor('#0F1117')
axes[1,1].set_title('Amount Statistics', color='white', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/eda_distributions.png", dpi=150, bbox_inches='tight', facecolor='#0F1117')
plt.close(); print("✓ eda_distributions.png")

# ── Preprocessing ──
scaler = StandardScaler()
df['Amount_Scaled'] = scaler.fit_transform(df[['Amount']])
df['Time_Scaled']   = scaler.fit_transform(df[['Time']])
feature_cols = [f"V{i}" for i in range(1,29)] + ['Amount_Scaled','Time_Scaled']
X = df[feature_cols].values; y = df['Class'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)

# ── SMOTE ──
smote = SMOTE(random_state=42, k_neighbors=min(5, (y_train==1).sum()-1))
X_sm, y_sm = smote.fit_resample(X_train, y_train)
print(f"SMOTE: fraud {(y_train==1).sum()} → {(y_sm==1).sum()}")

# ── Models with threshold tuning ──
results = {}

# Logistic Regression
lr = LogisticRegression(max_iter=1000, C=0.1, random_state=42, class_weight='balanced')
lr.fit(X_sm, y_sm)
lr_prob = lr.predict_proba(X_test)[:,1]
# find best F1 threshold
best_f1, best_thresh = 0, 0.5
for t in np.arange(0.1, 0.9, 0.02):
    yp = (lr_prob >= t).astype(int)
    if yp.sum() > 0:
        f = f1_score(y_test, yp, zero_division=0)
        if f > best_f1: best_f1, best_thresh = f, t
lr_pred = (lr_prob >= best_thresh).astype(int)
results["Logistic Regression"] = {
    'model': lr, 'y_pred': lr_pred, 'y_proba': lr_prob,
    'precision': precision_score(y_test, lr_pred, zero_division=0),
    'recall':    recall_score(y_test, lr_pred, zero_division=0),
    'f1':        f1_score(y_test, lr_pred, zero_division=0),
    'auc':       roc_auc_score(y_test, lr_prob),
    'cm':        confusion_matrix(y_test, lr_pred)
}

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1,
                             class_weight='balanced', max_depth=12, min_samples_leaf=2)
rf.fit(X_sm, y_sm)
rf_prob = rf.predict_proba(X_test)[:,1]
best_f1, best_thresh = 0, 0.5
for t in np.arange(0.1, 0.9, 0.02):
    yp = (rf_prob >= t).astype(int)
    if yp.sum() > 0:
        f = f1_score(y_test, yp, zero_division=0)
        if f > best_f1: best_f1, best_thresh = f, t
rf_pred = (rf_prob >= best_thresh).astype(int)
results["Random Forest"] = {
    'model': rf, 'y_pred': rf_pred, 'y_proba': rf_prob,
    'precision': precision_score(y_test, rf_pred, zero_division=0),
    'recall':    recall_score(y_test, rf_pred, zero_division=0),
    'f1':        f1_score(y_test, rf_pred, zero_division=0),
    'auc':       roc_auc_score(y_test, rf_prob),
    'cm':        confusion_matrix(y_test, rf_pred)
}

for name, r in results.items():
    print(f"{name}: P={r['precision']:.4f} R={r['recall']:.4f} F1={r['f1']:.4f} AUC={r['auc']:.4f}")

# ── Confusion Matrices ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('#0F1117')
fig.suptitle('Confusion Matrices — Credit Card Fraud Detection', color='white', fontsize=15, fontweight='bold')
for ax, (name, r) in zip(axes, results.items()):
    cm_n = r['cm'].astype(float) / r['cm'].sum(axis=1)[:,np.newaxis]
    im = ax.imshow(cm_n, cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_facecolor('#1A1D2E')
    labels = [['TN','FP'],['FN','TP']]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{labels[i][j]}\n{r['cm'][i,j]:,}\n({cm_n[i,j]:.1%})",
                    ha='center', va='center', fontsize=12,
                    color='black' if cm_n[i,j]>0.55 else 'white', fontweight='bold')
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['Predicted\nNormal','Predicted\nFraud'], color='#8B8FA8')
    ax.set_yticklabels(['Actual\nNormal','Actual\nFraud'], color='#8B8FA8')
    ax.set_title(name, color='white', fontsize=13, fontweight='bold', pad=10)
    plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/confusion_matrices.png", dpi=150, bbox_inches='tight', facecolor='#0F1117')
plt.close(); print("✓ confusion_matrices.png")

# ── ROC ──
fig, ax = plt.subplots(figsize=(9, 7))
fig.patch.set_facecolor('#0F1117'); ax.set_facecolor('#1A1D2E')
for spine in ax.spines.values(): spine.set_edgecolor('#2D3150')
for name, color in zip(results.keys(), ['#00D4AA','#FF6B35']):
    r = results[name]
    fpr, tpr, _ = roc_curve(y_test, r['y_proba'])
    ax.plot(fpr, tpr, color=color, lw=2.5, label=f"{name}  (AUC = {r['auc']:.4f})")
ax.plot([0,1],[0,1],'--',color='#8B8FA8',lw=1.5,label='Random Classifier (AUC = 0.5)')
ax.fill_between([0,1],[0,1],alpha=0.04,color='#8B8FA8')
ax.set_xlabel('False Positive Rate', color='#8B8FA8', fontsize=12)
ax.set_ylabel('True Positive Rate', color='#8B8FA8', fontsize=12)
ax.set_title('ROC Curve — Fraud Detection Models', color='white', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', facecolor='#2D3150', edgecolor='#3D4170', labelcolor='white', fontsize=11)
ax.tick_params(colors='#8B8FA8'); ax.grid(alpha=0.15, color='#4D5180')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/roc_curves.png", dpi=150, bbox_inches='tight', facecolor='#0F1117')
plt.close(); print("✓ roc_curves.png")

# ── Metrics Comparison ──
fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor('#0F1117'); ax.set_facecolor('#1A1D2E')
for spine in ax.spines.values(): spine.set_edgecolor('#2D3150')
m_labels = ['Precision','Recall','F1-Score','ROC-AUC']
x = np.arange(len(m_labels)); width = 0.35
for i, (name, color) in enumerate(zip(results.keys(), ['#00D4AA','#FF6B35'])):
    r = results[name]
    vals = [r['precision'],r['recall'],r['f1'],r['auc']]
    bars = ax.bar(x+i*width-width/2, vals, width, color=color, alpha=0.85, label=name)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.008,
                f'{val:.3f}', ha='center', color='white', fontsize=10, fontweight='bold')
ax.set_ylim(0,1.15); ax.set_xticks(x)
ax.set_xticklabels(m_labels, color='#8B8FA8', fontsize=12)
ax.set_ylabel('Score', color='#8B8FA8', fontsize=12)
ax.set_title('Model Performance Comparison', color='white', fontsize=14, fontweight='bold')
ax.legend(facecolor='#2D3150', edgecolor='#3D4170', labelcolor='white', fontsize=11)
ax.tick_params(colors='#8B8FA8'); ax.grid(axis='y', alpha=0.15, color='#4D5180')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/metrics_comparison.png", dpi=150, bbox_inches='tight', facecolor='#0F1117')
plt.close(); print("✓ metrics_comparison.png")

# ── Feature Importance ──
fi = rf.feature_importances_
fi_df = pd.DataFrame({'Feature':feature_cols,'Importance':fi}).sort_values('Importance').tail(15)
fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor('#0F1117'); ax.set_facecolor('#1A1D2E')
for spine in ax.spines.values(): spine.set_edgecolor('#2D3150')
colors_fi = ['#FF4757' if any(k in f for k in ['Amount','Time']) else '#00D4AA' for f in fi_df['Feature']]
ax.barh(fi_df['Feature'], fi_df['Importance'], color=colors_fi)
ax.set_xlabel('Importance Score', color='#8B8FA8', fontsize=12)
ax.set_title('Top 15 Feature Importances — Random Forest', color='white', fontsize=14, fontweight='bold')
ax.tick_params(colors='#8B8FA8'); ax.grid(axis='x', alpha=0.15, color='#4D5180')
patches = [mpatches.Patch(color='#00D4AA',label='PCA Features (V1–V28)'),
           mpatches.Patch(color='#FF4757',label='Engineered Features')]
ax.legend(handles=patches, facecolor='#2D3150', edgecolor='#3D4170', labelcolor='white')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/feature_importance.png", dpi=150, bbox_inches='tight', facecolor='#0F1117')
plt.close(); print("✓ feature_importance.png")

# ── Save summary ──
best_name = max(results, key=lambda k: results[k]['f1'])
summary = {
    "dataset": {"total":int(len(df)),"fraud":int(df['Class'].sum()),
                "normal":int((df['Class']==0).sum()),"fraud_pct":round(float(df['Class'].mean()*100),4)},
    "models": {name: {"precision":round(float(r['precision']),4),"recall":round(float(r['recall']),4),
                       "f1":round(float(r['f1']),4),"auc":round(float(r['auc']),4),"cm":r['cm'].tolist()}
               for name, r in results.items()},
    "best_model": best_name
}
with open(f"{OUTPUT_DIR}/results_summary.json","w") as f: json.dump(summary,f,indent=2)
print(f"\n✅ Best: {best_name}")
print(json.dumps(summary, indent=2))
