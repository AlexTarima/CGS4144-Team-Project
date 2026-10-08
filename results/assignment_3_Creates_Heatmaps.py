from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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
#keep = metadata[metadata["refinebio_disease"].isin(["tb subjects", "healthy controls", "mtp controls", "lung dx controls"])].copy()
#keep["group"] = keep["refinebio_disease"].map({"tb subjects": "TB", "healthy controls": "Controls", "mtp controls" : "Controls", "lung dx controls" : "Controls"})

keep = metadata[metadata["refinebio_disease"].isin(["tb subjects", "healthy controls"])].copy()
keep["group"] = keep["refinebio_disease"].map({"tb subjects": "TB", "healthy controls": "Controls"})
print(keep["group"].value_counts())    # expect TB 741, Controls 154

log_mat = log_mat[keep.index]          # 895 samples
metadata = keep

# ---- Variance ranking on the filtered samples (no group labels used) ----
gene_var = log_mat.var(axis=1)

def top_genes(n):
    return log_mat.loc[gene_var.nlargest(n).index]

# ---- K-means across k ----
X = top_genes(5000).T                  # samples x genes
print(X.shape)                         # expect (895, 5000)

results, rows = {}, []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    results[k] = km.labels_
    rows.append({"k": k, "inertia": km.inertia_,
                 "silhouette": silhouette_score(X, km.labels_)})

scores = pd.DataFrame(rows)
print(scores)

from itertools import combinations
from scipy.stats import chi2_contingency, false_discovery_control
from sklearn.metrics import adjusted_rand_score

gene_counts = [10, 100, 1000, 5000, 10000]
k_values = range(2, 9)

# k used for the gene-count comparison. Silhouette often picks 2, so
# choose it deliberately and say why in the write-up.
k_main = int(scores.loc[scores["silhouette"].idxmax(), "k"])
print("k_main =", k_main)

# ---- Cluster at every (gene count, k) ----
all_labels = {}
for n in gene_counts:
    Xn = top_genes(n).T                       # samples x genes
    for k in k_values:
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Xn)
        all_labels[(n, k)] = pd.Series(km.labels_, index=Xn.index)

def chi2_row(a, b, name_a, name_b):
    table = pd.crosstab(a, b)
    chi2, p, dof, expected = chi2_contingency(table)
    return {"comparison": f"{name_a} vs {name_b}", "chi2": chi2, "dof": dof,
            "p_value": p, "pct_cells_expected_lt5": (expected < 5).mean() * 100}

rows = []

# (e.ii) Each pair of gene counts, at k_main
for n1, n2 in combinations(gene_counts, 2):
    r = chi2_row(all_labels[(n1, k_main)], all_labels[(n2, k_main)],
                 f"{n1} genes (k={k_main})", f"{n2} genes (k={k_main})")
    r["type"] = "gene count vs gene count"
    r["ARI"] = adjusted_rand_score(all_labels[(n1, k_main)], all_labels[(n2, k_main)])
    rows.append(r)

# (Step 4) Every clustering result vs the Assignment 1 groups
for (n, k), lab in all_labels.items():
    r = chi2_row(lab, metadata.loc[lab.index, "group"],
                 f"{n} genes, k={k}", "TB/Healthy")
    r["type"] = "clusters vs groups"
    r["ARI"] = adjusted_rand_score(lab, metadata.loc[lab.index, "group"])
    rows.append(r)

chi_table = pd.DataFrame(rows)
chi_table["adj_p_BH"] = false_discovery_control(chi_table["p_value"], method="bh")
chi_table = chi_table[["type", "comparison", "chi2", "dof", "p_value",
                       "adj_p_BH", "ARI", "pct_cells_expected_lt5"]]

Path("results").mkdir(exist_ok=True)
chi_table.to_csv("results/kmeans_chisq_table.tsv", sep="\t", index=False)
print(chi_table.to_string())


for n in (10, 1000, 5000):
    print(f"\n{n} genes, k=2")
    print(pd.crosstab(all_labels[(n, 2)], metadata.loc[all_labels[(n, 2)].index, "group"]))



#HEATMAP

import seaborn as sns
import matplotlib.pyplot as plt

k_chosen = 2
X5 = top_genes(5000)               # genes x samples

pam = pd.read_csv("results/pam_labels.tsv", sep="\t", index_col="sample")

pam_col = "pam_g5000_k2"        # use the name the R script printed at the end
pam_labels = pam[pam_col].map(lambda c: f"P{c}")   # P1, P2 so it isn't confused with C0, C1

gmm = pd.read_csv("results/gmm_labels.tsv", sep="\t", index_col="sample")
gmm_col = "gmm_g5000_k2"        # use the name the R script printed at the end
gmm_labels = gmm[gmm_col].map(lambda c: f"P{c}")   # P1, P2 so it isn't confused with C0, C1


ann = pd.DataFrame({
    "k-means (k=2)": all_labels[(5000, k_chosen)].map(lambda c: f"C{c}"),
    "PAM (k=2)": pam_labels.reindex(X5.columns),
    "GMM (k=2)": gmm_labels.reindex(X5.columns),
    "Group": metadata.loc[X5.columns, "group"],
})

print(ann.isna().sum())   # should be all zeros

def colors_for(series, palette):
    lut = dict(zip(sorted(series.unique()), sns.color_palette(palette, series.nunique())))
    return series.map(lut), lut

col_colors, luts = {}, {}
palettes = ["Set1", "Set2", "Dark2"]
for i, c in enumerate(ann.columns):
    col_colors[c], luts[c] = colors_for(ann[c], palettes[i % 3])
col_colors = pd.DataFrame(col_colors)

g = sns.clustermap(
    X5, z_score=0, cmap="vlag", center=0, vmin=-3, vmax=3,
    method="average", metric="euclidean",
    col_colors=col_colors,
    xticklabels=False, yticklabels=False,
    figsize=(12, 10),
    cbar_kws={"label": "Expression (z-score)"},
)
g.ax_heatmap.set_xlabel("Samples (n = 819)")
g.ax_heatmap.set_ylabel("Genes (top 5,000 most variable)")

# legends for the annotation strips
for c, lut in luts.items():
    handles = [plt.Rectangle((0, 0), 1, 1, color=col) for col in lut.values()]
    g.ax_col_dendrogram.legend(handles, lut.keys(), title=c, loc="upper left",
                               bbox_to_anchor=(1.02 + 0.2 * list(luts).index(c), 1.0))
g.savefig("results/heatmap_5000_genes.png", dpi=200, bbox_inches="tight")