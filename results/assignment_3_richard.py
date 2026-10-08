from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, false_discovery_control
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

data_dir = Path("data/SRP092402")
results_dir = Path("results")

SEED = 42
MAX_PCS = 20
GENE_COUNTS = [10, 100, 1000, 5000, 10000]
K_VALUES = range(2, 9)


def label_name(n, k):
    return f"gmm_g{n}_k{k}"


expression_mat = pd.read_csv(data_dir / "SRP092402.tsv", sep="\t", index_col="Gene")
metadata = pd.read_csv(data_dir / "metadata_SRP092402.tsv", sep="\t", index_col="refinebio_accession_code")

metadata = metadata[metadata["refinebio_disease"].isin(["tb subjects", "healthy controls"])].copy()
metadata["group"] = np.where(metadata["refinebio_disease"] == "tb subjects", "TB", "Healthy")
print(metadata["group"].value_counts())

log_mat = np.log2(expression_mat[metadata.index] + 1)
gene_order = log_mat.var(axis=1).sort_values(ascending=False).index

labels, score_rows = {}, []
for n in GENE_COUNTS:
    X = log_mat.loc[gene_order[:n]].T
    X = X - X.mean()

    X_pcs = PCA(n_components=min(MAX_PCS, n), random_state=SEED).fit_transform(X)
    for k in K_VALUES:
        gmm = GaussianMixture(n_components=k, covariance_type="full", n_init=10, random_state=SEED).fit(X_pcs)
        lab = pd.Series(gmm.predict(X_pcs) + 1, index=X.index)
        labels[label_name(n, k)] = lab
        score_rows.append({"genes": n, "k": k, "bic": gmm.bic(X_pcs),
                           "cluster_sizes": "/".join(map(str, lab.value_counts().sort_index()))})

scores = pd.DataFrame(score_rows)
scores.to_csv(results_dir / "gmm_bic_by_k.tsv", sep="\t", index=False)
print(scores.to_string(index=False))

s5000 = scores[scores["genes"] == 5000]
k_main = int(s5000.loc[s5000["bic"].idxmin(), "k"])
print(f"k_main = {k_main}")

def chi2_row(a, b, name_a, name_b, comparison_type):
    table = pd.crosstab(a, b)
    chi2, p, dof, expected = chi2_contingency(table) 
    return {"type": comparison_type, "comparison": f"{name_a} vs {name_b}", "chi2": chi2, "dof": dof,
            "p_value": p, "ARI": adjusted_rand_score(a, b),
            "pct_cells_expected_lt5": (expected < 5).mean() * 100}


rows = []
for k1, k2 in zip(K_VALUES, K_VALUES[1:]):
    rows.append(chi2_row(labels[label_name(5000, k1)], labels[label_name(5000, k2)],
                         f"5000 genes (k={k1})", f"5000 genes (k={k2})", "k vs k"))
for n1, n2 in combinations(GENE_COUNTS, 2):
    rows.append(chi2_row(labels[label_name(n1, k_main)], labels[label_name(n2, k_main)],
                         f"{n1} genes (k={k_main})", f"{n2} genes (k={k_main})", "gene count vs gene count"))
for n in GENE_COUNTS:
    for k in K_VALUES:
        lab = labels[label_name(n, k)]
        rows.append(chi2_row(lab, metadata.loc[lab.index, "group"], f"{n} genes, k={k}", "TB/Healthy",
                             "clusters vs groups"))

chi_table = pd.DataFrame(rows)
chi_table.insert(0, "method", "GMM")
chi_table["adj_p_BH"] = false_discovery_control(chi_table["p_value"], method="bh")
chi_table = chi_table[["method", "type", "comparison", "chi2", "dof", "p_value",
                       "adj_p_BH", "ARI", "pct_cells_expected_lt5"]]
chi_table.to_csv(results_dir / "gmm_chisq_table.tsv", sep="\t", index=False)
print(chi_table.to_string(index=False))

with open(results_dir / "gmm_crosstabs.txt", "w") as f:
    for k1, k2 in zip(K_VALUES, K_VALUES[1:]):
        f.write(f"\n5000 genes: k = {k1} (rows) vs k = {k2} (columns)\n")
        f.write(pd.crosstab(labels[label_name(5000, k1)], labels[label_name(5000, k2)]).to_string() + "\n")
    for n in GENE_COUNTS:
        lab = labels[label_name(n, k_main)]
        f.write(f"\n{n} genes, k = {k_main}: clusters vs TB/Healthy\n")
        f.write(pd.crosstab(lab.rename("cluster"), metadata.loc[lab.index, "group"]).to_string() + "\n")

labels_out = pd.DataFrame(labels)
labels_out.insert(0, "group", metadata.loc[labels_out.index, "group"])
labels_out.index.name = "sample"
labels_out.to_csv(results_dir / "gmm_labels.tsv", sep="\t")
print(f"Heatmap column for the team: {label_name(5000, k_main)}")
