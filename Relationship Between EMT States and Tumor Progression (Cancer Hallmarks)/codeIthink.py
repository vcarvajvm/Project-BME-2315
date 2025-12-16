# ==== Setup & load (same style as example_EDA.py) ====
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression

# data paths match class repo layout used in example_EDA.py
expr = pd.read_csv('GSE62944_subsample_topVar_log2TPM.csv', index_col=0)
meta = pd.read_csv('GSE62944_metadata.csv', index_col=0)

# Pick a cancer type with decent metadata coverage (you can change this)
samples = meta.index
X = expr[samples]    # genes x samples
M = meta.loc[samples]

# ==== Define invasion/metastasis gene sets ====
epithelial = ['CDH1','EPCAM','CLDN1','KRT8','KRT18']
mesenchymal = ['VIM','SNAI1','SNAI2','TWIST1','ZEB1','ZEB2','MMP2','MMP9','ITGA5','ITGB1','CXCR4','COL1A1','FN1']

# keep only genes present
epi_present = [g for g in epithelial if g in X.index]
mes_present = [g for g in mesenchymal if g in X.index]

# ==== Compute EMT score per sample ====
mes_score = X.loc[mes_present].mean(axis=0)
epi_score = X.loc[epi_present].mean(axis=0)
emt_score = mes_score - epi_score
emt_df = pd.DataFrame({'EMT_score': emt_score})
merged = emt_df.join(M, how='left')

# ==== Unsupervised: visualize ====
plt.figure()
merged['EMT_score'].plot.hist(bins=30)
plt.title(f'EMT score distribution')
plt.xlabel('EMT_score'); plt.ylabel('Count')
plt.show()

# Compare EMT by stage if available
# --- Compare EMT by stage if available (robust version) ---
# Try a broader set of common stage field names
stage_candidates = [
    'pathologic_stage', 'ajcc_pathologic_tumor_stage', 'clinical_stage',
    'ajcc_pathologic_stage', 'tumor_stage'
]
# keep only those present in the dataframe
stage_candidates = [c for c in stage_candidates if c in merged.columns]

# pick the first column that actually has >1 unique, non-null values
valid_stage_cols = []
for c in stage_candidates:
    nn = merged[c].notna().sum()
    nu = merged[c].nunique(dropna=True)
    if nn > 0 and nu > 1:
        valid_stage_cols.append(c)

if valid_stage_cols:
    col = valid_stage_cols[0]
    # clean up the stage labels a bit for nicer grouping
    tmp = merged[['EMT_score', col]].dropna().copy()
    tmp[col] = (
        tmp[col].astype(str)
        .str.upper()
        .str.replace('STAGE ', '', regex=False)
        .str.replace('AJCC ', '', regex=False)
        .str.replace('TUMOR ', '', regex=False)
        .str.replace('PATHOLOGIC ', '', regex=False)
        .str.strip()
    )

    print(f"Using grouping column: {col}")
    print("Top levels:\n", tmp[col].value_counts().head())

    plt.figure()
    tmp.boxplot(column='EMT_score', by=col, rot=45)
    plt.title(f'EMT score by {col} in {CANCER}')
    plt.suptitle('')
    plt.xlabel(col); plt.ylabel('EMT_score')
    plt.show()
else:
    # helpful debug info so you can choose another variable to group on
    print("⚠️ No usable stage column found. Here are candidates with non-null counts and nunique:")
    for c in stage_candidates:
        print(f"{c}: non-null={merged[c].notna().sum()}  unique={merged[c].nunique(dropna=True)}")
    print("Try grouping by another metadata field with coverage, e.g. 'sample_type' or 'cancer_type'.")


# PCA on panel (z-scored across samples)
panel_genes = list(set(epi_present + mes_present))
scaler = StandardScaler(with_mean=True, with_std=True)
Z = scaler.fit_transform(X.loc[panel_genes].T)  # samples x genes
pca = PCA(n_components=2).fit(Z)
PC = pca.transform(Z)
pc_df = pd.DataFrame(PC, index=samples, columns=['PC1','PC2']).join(emt_df)

plt.figure()
plt.scatter(pc_df['PC1'], pc_df['PC2'], c=pc_df['EMT_score'], cmap='viridis')
plt.title(f'{CANCER} PCA (panel genes), colored by EMT_score')
plt.xlabel('PC1'); plt.ylabel('PC2')
plt.colorbar(label='EMT_score')
plt.show()

# ==== Supervised: predict metastasis-related label ====
# Try to find a metastasis-relevant label with decent coverage
candidate_labels = ['pathologic_M','ajcc_pathologic_m','metastatic_at_diagnosis',
                    'pathologic_N','ajcc_pathologic_n','lymph_nodes_positive']
label = next((c for c in candidate_labels if c in merged.columns and merged[c].notna().sum()>0), None)

if label:
    yraw = merged[label].astype(str).str.upper()
    # binarize: M1 vs not; for N, N+ vs N0; for yes/no fields
    if 'M' in label.upper():
        y = (yraw.str.contains('1')).astype(int)
    elif 'N' in label.upper():
        y = (~yraw.str.contains('0')).astype(int)  # any N+ → 1
    else:
        y = yraw.isin(['YES','TRUE','1']).astype(int)

    feats = pd.DataFrame({'EMT_score': emt_df['EMT_score']}).dropna()
    y = y.loc[feats.index].dropna()
    idx = feats.index.intersection(y.index)
    Xtr, Xte, ytr, yte = train_test_split(feats.loc[idx], y.loc[idx], test_size=0.3, random_state=42, stratify=y.loc[idx])

    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    proba = clf.predict_proba(Xte)[:,1]
    print('AUC:', roc_auc_score(yte, proba))
    print(confusion_matrix(yte, pred))
    print(classification_report(yte, pred, digits=3))
else:
    print("No high-coverage metastasis-related label found in metadata for this cancer; report unsupervised results and/or switch cancer type.")
