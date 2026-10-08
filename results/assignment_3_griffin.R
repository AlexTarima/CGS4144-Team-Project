library(cluster)
 
dir.create("results", showWarnings = FALSE)
 
expression_df <- read.delim("data/SRP092402.tsv", row.names = 1, check.names = FALSE)
metadata <- read.delim("data/metadata_SRP092402.tsv", check.names = FALSE)
rownames(metadata) <- metadata$refinebio_accession_code
 
log_mat <- log2(as.matrix(expression_df) + 1)
 
print(table(metadata$refinebio_disease))
keep <- metadata[metadata$refinebio_disease %in% c("tb subjects", "healthy controls"), ]
keep$group <- ifelse(keep$refinebio_disease == "tb subjects", "TB", "Healthy")
keep <- keep[rownames(keep) %in% colnames(log_mat), ]
log_mat <- log_mat[, rownames(keep)]
metadata <- keep
print(table(metadata$group))
 
gene_var <- apply(log_mat, 1, var)
gene_order <- names(sort(gene_var, decreasing = TRUE))
top_genes <- function(n) log_mat[gene_order[seq_len(n)], , drop = FALSE]
 
gene_counts <- c(10, 100, 1000, 5000, 10000)
k_values <- 2:8
label_name <- function(n, k) paste0("pam_g", n, "_k", k)
 
all_labels <- list()
score_rows <- list()
for (n in gene_counts) {
  d <- dist(t(top_genes(n)), method = "euclidean")
  for (k in k_values) {
    fit <- pam(d, k = k, diss = TRUE)
    name <- label_name(n, k)
    all_labels[[name]] <- setNames(fit$clustering, colnames(log_mat))
    score_rows[[name]] <- data.frame(
      genes = n,
      k = k,
      avg_silhouette = fit$silinfo$avg.width,
      cluster_sizes = paste(as.vector(table(fit$clustering)), collapse = "/")
    )
    cat("done:", name, "\n")
  }
}
 
scores <- do.call(rbind, score_rows)
rownames(scores) <- NULL
write.table(scores, "results/pam_silhouette_by_k.tsv", sep = "\t", quote = FALSE, row.names = FALSE)
print(scores)
 
s5000 <- scores[scores$genes == 5000, ]
k_main <- s5000$k[which.max(s5000$avg_silhouette)]
cat("k_main =", k_main, "\n")
 
ari <- function(a, b) {
  tab <- table(a, b)
  pairs <- function(x) sum(choose(x, 2))
  index <- pairs(tab)
  row_pairs <- pairs(rowSums(tab))
  col_pairs <- pairs(colSums(tab))
  expected <- row_pairs * col_pairs / choose(sum(tab), 2)
  (index - expected) / ((row_pairs + col_pairs) / 2 - expected)
}
 
chi2_row <- function(a, b, name_a, name_b, type) {
  tab <- table(a, b)
  test <- suppressWarnings(chisq.test(tab))
  data.frame(
    type = type,
    comparison = paste(name_a, "vs", name_b),
    chi2 = unname(test$statistic),
    dof = unname(test$parameter),
    p_value = test$p.value,
    ARI = ari(a, b),
    pct_cells_expected_lt5 = mean(test$expected < 5) * 100
  )
}
 
rows <- list()
 
for (i in seq_len(length(k_values) - 1)) {
  k1 <- k_values[i]
  k2 <- k_values[i + 1]
  rows[[length(rows) + 1]] <- chi2_row(
    all_labels[[label_name(5000, k1)]], all_labels[[label_name(5000, k2)]],
    paste0("5000 genes (k=", k1, ")"), paste0("5000 genes (k=", k2, ")"),
    "k vs k"
  )
}
 
gene_pairs <- combn(gene_counts, 2)
for (j in seq_len(ncol(gene_pairs))) {
  n1 <- gene_pairs[1, j]
  n2 <- gene_pairs[2, j]
  rows[[length(rows) + 1]] <- chi2_row(
    all_labels[[label_name(n1, k_main)]], all_labels[[label_name(n2, k_main)]],
    paste0(n1, " genes (k=", k_main, ")"), paste0(n2, " genes (k=", k_main, ")"),
    "gene count vs gene count"
  )
}
 
for (n in gene_counts) {
  for (k in k_values) {
    lab <- all_labels[[label_name(n, k)]]
    rows[[length(rows) + 1]] <- chi2_row(
      lab, metadata[names(lab), "group"],
      paste0(n, " genes, k=", k), "TB/Healthy",
      "clusters vs groups"
    )
  }
}
 
chi_table <- do.call(rbind, rows)
chi_table$method <- "PAM"
chi_table$adj_p_BH <- p.adjust(chi_table$p_value, method = "BH")
chi_table <- chi_table[, c("method", "type", "comparison", "chi2", "dof", "p_value",
                           "adj_p_BH", "ARI", "pct_cells_expected_lt5")]
write.table(chi_table, "results/pam_chisq_table.tsv", sep = "\t", quote = FALSE, row.names = FALSE)
print(chi_table)
 
sink("results/pam_crosstabs.txt")
for (i in seq_len(length(k_values) - 1)) {
  cat("\n5000 genes: k =", k_values[i], "(rows) vs k =", k_values[i + 1], "(columns)\n")
  print(table(all_labels[[label_name(5000, k_values[i])]],
              all_labels[[label_name(5000, k_values[i + 1])]]))
}
for (n in gene_counts) {
  lab <- all_labels[[label_name(n, k_main)]]
  cat("\n", n, " genes, k = ", k_main, ": clusters vs TB/Healthy\n", sep = "")
  print(table(cluster = lab, group = metadata[names(lab), "group"]))
}
sink()
 
labels_out <- data.frame(sample = colnames(log_mat), group = metadata[colnames(log_mat), "group"],
                         as.data.frame(all_labels), check.names = FALSE)
write.table(labels_out, "results/pam_labels.tsv", sep = "\t", quote = FALSE, row.names = FALSE)
cat("Heatmap column for the team: ", label_name(5000, k_main), "\n", sep = "")
