# Gene Ontology enrichment of the TB vs healthy differentially expressed genes with topGO
# Method: topGO, Fisher's exact test with the weight01 algorithm
# Ontology: Gene Ontology, Biological Process (BP)
# Run from the project root after "results/S3. Volcano Graph.ipynb": Rscript topgo_enrichment.R

# Install packages if needed
if (!("BiocManager" %in% installed.packages())) {
  install.packages("BiocManager", repos = "https://cloud.r-project.org")
}
for (pkg in c("topGO", "org.Hs.eg.db")) {
  if (!(pkg %in% installed.packages())) {
    BiocManager::install(pkg, update = FALSE)
  }
}
for (pkg in c("magrittr", "readr", "dplyr")) {
  if (!(pkg %in% installed.packages())) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
}

suppressPackageStartupMessages({
  library(topGO)
  library(org.Hs.eg.db)
  library(magrittr)
})

# File paths
results_dir <- "results"
de_file <- file.path(results_dir, "diff_expr_tb_vs_healthy.tsv")
output_file <- file.path(results_dir, "topgo_GO_BP_tb_vs_healthy.tsv")

# Significance cutoff on the weight01 p-value. topGO's authors advise against
# multiple-testing correction for weight01, because its tests are not independent.
PVALUE_CUTOFF <- 0.01

# Read the differential expression results
de_results <- readr::read_tsv(de_file, show_col_types = FALSE)

# Gene universe: every gene that was tested. 1 = significantly differentially expressed.
all_genes <- factor(as.integer(de_results$direction != "Not significant"))
names(all_genes) <- de_results$Ensembl
message(
  "Universe: ", length(all_genes), " genes; significant: ", sum(all_genes == 1)
)

# Build the topGO object, annotating Ensembl IDs to GO terms with org.Hs.eg.db
go_data <- new(
  "topGOdata",
  description = "TB at diagnosis vs healthy controls",
  ontology = "BP",
  allGenes = all_genes,
  annot = annFUN.org,
  mapping = "org.Hs.eg.db",
  ID = "ensembl",
  nodeSize = 10 # skip GO terms with fewer than 10 annotated genes
)
go_data

# weight01: topGO's default; accounts for the GO hierarchy so parent terms aren't
# significant only because of their children. classic: plain Fisher test per term.
weight01_test <- runTest(go_data, algorithm = "weight01", statistic = "fisher")
classic_test <- runTest(go_data, algorithm = "classic", statistic = "fisher")

n_terms <- length(usedGO(go_data))
go_table <- GenTable(
  go_data,
  weight01 = weight01_test,
  classic = classic_test,
  orderBy = "weight01",
  topNodes = n_terms,
  numChar = 1000
)

# GenTable returns p-values as text (e.g. "< 1e-30"); convert back to numbers
go_table <- go_table %>%
  dplyr::mutate(
    pvalue_weight01 = score(weight01_test)[GO.ID],
    pvalue_classic = score(classic_test)[GO.ID],
    padj_classic_BH = p.adjust(pvalue_classic, method = "BH"),
    fold_enrichment = Significant / Expected,
    significant = pvalue_weight01 < PVALUE_CUTOFF
  )

# List the significant genes (as symbols) in each term, split by direction
sig_de <- de_results %>% dplyr::filter(direction != "Not significant")
sig_genes_in_terms <- genesInTerm(go_data, go_table$GO.ID) %>%
  lapply(function(genes) sig_de[sig_de$Ensembl %in% genes, ])

label_genes <- function(df) {
  ifelse(is.na(df$symbol), df$Ensembl, df$symbol) %>% paste(collapse = ";")
}
go_table$n_higher_in_TB <- sapply(sig_genes_in_terms, function(df) sum(df$direction == "Higher in TB"))
go_table$n_lower_in_TB <- sapply(sig_genes_in_terms, function(df) sum(df$direction == "Lower in TB"))
go_table$genes_higher_in_TB <- sapply(sig_genes_in_terms, function(df) label_genes(df[df$direction == "Higher in TB", ]))
go_table$genes_lower_in_TB <- sapply(sig_genes_in_terms, function(df) label_genes(df[df$direction == "Lower in TB", ]))

go_table <- go_table %>%
  dplyr::select(
    GO_ID = GO.ID, term = Term, annotated = Annotated, significant_genes = Significant,
    expected = Expected, fold_enrichment, pvalue_weight01, pvalue_classic, padj_classic_BH,
    significant, n_higher_in_TB, n_lower_in_TB, genes_higher_in_TB, genes_lower_in_TB
  ) %>%
  dplyr::arrange(pvalue_weight01)

readr::write_tsv(go_table, output_file)
message(
  "Tested ", nrow(go_table), " GO terms; ", sum(go_table$significant),
  " significant (weight01 p < ", PVALUE_CUTOFF, ")"
)
message("Wrote ", output_file)
