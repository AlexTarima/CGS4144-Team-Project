from pathlib import Path
import numpy as np
import pandas as pd
import sns
from sklearn.pipeline import make_pipeline

# ---- Load ----
expression_mat = pd.read_csv("data/SRP092402.tsv", sep="\t", index_col="Gene")
expression_mat.index.name = "Ensembl"
metadata = pd.read_csv("data/metadata_SRP092402.tsv", sep="\t",
                       index_col="refinebio_accession_code")

# symbol lookup (labels only), reusing the Assignment 2 DE table
sym = (pd.read_csv("data/diff_expr_tb_vs_healthy.tsv", sep="\t")
         .drop_duplicates("Ensembl").set_index("Ensembl")["symbol"])
symbols = pd.Series(expression_mat.index.map(sym), index=expression_mat.index)

# ---- Log-scale ----
log_mat = np.log2(expression_mat + 1)

# ---- Filter samples to the two Assignment 1 groups ----
print(metadata["refinebio_disease"].value_counts())
keep = metadata[metadata["refinebio_disease"].isin(["tb subjects", "healthy controls", "mtp controls", "lung dx controls"])].copy()
keep["group"] = keep["refinebio_disease"].map({"tb subjects": "TB", "healthy controls": "Controls", "mtp controls" : "Controls", "lung dx controls" : "Controls"})

#keep = metadata[metadata["refinebio_disease"].isin(["tb subjects", "healthy controls"])].copy()
#keep["group"] = keep["refinebio_disease"].map({"tb subjects": "TB", "healthy controls": "Controls"})
print(keep["group"].value_counts())    # expect TB 741, Controls 154

log_mat = log_mat[keep.index]          # 895 samples
metadata = keep

# ---- Variance ranking on the filtered samples (no group labels used) ----
gene_var = log_mat.var(axis=1)

def top_genes(n):
    return log_mat.loc[gene_var.nlargest(n).index]

# ---- K-means across k ----
#X = top_genes(5000).T                  # samples x genes
#print(X.shape)                         # expect (895, 5000)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

np.random.seed(0)

# SVM TRAINING, 2a-c
X = top_genes(5000).T
print(X.shape)
y = metadata.loc[X.index, "group"]  # "TB" vs "Controls"
assert (X.index == y.index).all()  # rows must line up
# clustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


results, rows = {}, []
k=2
km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
results[k] = km.labels_
rows.append({"k": k, "inertia": km.inertia_,
                "silhouette": silhouette_score(X, km.labels_)})

scores = pd.DataFrame(rows)
print(scores)

clusters = pd.Series(results[2], index=X.index, name="cluster")
print(clusters.value_counts())
y_group   = metadata.loc[X.index, "group"]    # Part 2d: TB vs Controls
y_cluster = clusters                          # Part 2e: k=2 cluster
assert (X.index == y_cluster.index).all()


def train_run_svm2(num_genes, seed=0):
    X = top_genes(num_genes).T
    print(X.shape)
    y = metadata.loc[X.index, "group"]    # "TB" vs "Controls"
    assert (X.index == y.index).all()     # rows must line up


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    svm_model = SVC(kernel="linear", C=1.0, probability=True, class_weight="balanced")

    svm_model.fit(X_train_scaled, y_train)

    y_pred = svm_model.predict(X_test_scaled)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}\n")
    print(classification_report(y_test, y_pred))

    Xn = top_genes(num_genes).T
    Xtr, Xte, ytr, yte = train_test_split(Xn, y.loc[Xn.index], test_size=0.2,
                                          random_state=seed, stratify=y.loc[Xn.index])
    m = make_pipeline(StandardScaler(),
                      SVC(kernel="linear", probability=True, class_weight="balanced"))
    m.fit(Xtr, ytr)
    proba = m.predict_proba(X_test)[:, 1]  # probability of cluster 1
    print("AUC:", roc_auc_score(yte, proba))
    return m, Xte, yte

def train_run_svm(num_genes, y, seed=0, verbose=True):
    X = top_genes(num_genes).T
    y = y.loc[X.index]

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y)

    model = make_pipeline(
        StandardScaler(),
        SVC(kernel="linear", C=1.0, probability=True, class_weight="balanced"))
    model.fit(Xtr, ytr)

    pos = model.classes_[-1]                      # "TB" for group; 1 for cluster
    proba = model.predict_proba(Xte)[:, list(model.classes_).index(pos)]
    auc = roc_auc_score(yte == pos, proba)

    if verbose:
        print(f"{num_genes} genes: shape {X.shape}, AUC {auc:.3f}")
        print(classification_report(yte, model.predict(Xte)))
    return {"n_genes": num_genes, "model": model, "Xte": Xte, "yte": yte, "auc": auc}

gene_counts = [10, 100, 1000, 5000, 10000]
res_group   = [train_run_svm(n, metadata["group"]) for n in gene_counts]
res_cluster = [train_run_svm(n, clusters) for n in gene_counts]   # clusters from the 5000-gene K-means

auc_table = pd.DataFrame({
    "n_genes": gene_counts,
    "AUC_group":   [r["auc"] for r in res_group],
    "AUC_cluster": [r["auc"] for r in res_cluster]})
print(auc_table)

def top_svm_genes(result, n=50):
    model = result["model"]
    svc = model.named_steps["svc"]
    genes = result["Xte"].columns                 # gene IDs, same order as coef_ columns
    coefs = pd.Series(svc.coef_[0], index=genes)
    return coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(n)

sig_group = top_svm_genes(res_group[3], n=50)     # index 3 = the 5000-gene model
print(sig_group.head(10))


# HEATMAP

signatures = {
    "SVM_group":   set(top_svm_genes(res_group[3], n=100).index),
    "SVM_cluster": set(top_svm_genes(res_cluster[3], n=100).index),
    # teammates add theirs, e.g.
    # "RF_group": set(...), "LR_group": set(...)
}

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# 1. Signature genes (Ensembl IDs), from the union step
signature_genes = sorted(set.union(*signatures.values()))

# 2. mat: genes x samples, z-scored per gene
mat = log_mat.loc[signature_genes]
mat = mat.sub(mat.mean(axis=1), axis=0).div(mat.std(axis=1), axis=0)

# 3. col_colors: Assignment 1 group for each sample (column of mat)
palette = {"TB": "tab:red", "Controls": "tab:blue"}
col_colors = metadata.loc[mat.columns, "group"].map(palette)

# 4. row_colors: which model(s) picked each gene
row_colors = pd.DataFrame(
    {name: ["black" if g in sig else "white" for g in signature_genes]
     for name, sig in signatures.items()},
    index=signature_genes)

# 5. The heatmap
g = sns.clustermap(
    mat,
    col_colors=col_colors,
    row_colors=row_colors,
    cmap="vlag", center=0, vmin=-3, vmax=3,
    xticklabels=False, yticklabels=False, figsize=(12, 12))


from pathlib import Path
Path("results").mkdir(exist_ok=True)
g.ax_col_dendrogram.legend(
    handles=[Patch(color=c, label=l) for l, c in palette.items()],
    title="Group", loc="center", ncol=2)
g.ax_heatmap.set_xlabel("Samples")
g.ax_heatmap.set_ylabel("Genes")

g.savefig("results/signature_heatmap.png", dpi=400, bbox_inches="tight")