# Convert Ensembl gene IDs in the SRP092402 expression matrix to gene symbols
# Based on the refine.bio example:
# https://alexslemonade.github.io/refinebio-examples/03-rnaseq/gene-id-annotation_rnaseq_01_ensembl.html
# Run from the project root: Rscript gene_id_annotation.R

# Install packages if needed
if (!("BiocManager" %in% installed.packages())) {
  install.packages("BiocManager", repos = "https://cloud.r-project.org")
}
if (!("org.Hs.eg.db" %in% installed.packages())) {
  BiocManager::install("org.Hs.eg.db", update = FALSE)
}
for (pkg in c("magrittr", "readr", "dplyr", "tibble", "tidyr")) {
  if (!(pkg %in% installed.packages())) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
}

library(org.Hs.eg.db)
library(magrittr)

# File paths
data_dir <- file.path("data", "SRP092402")
data_file <- file.path(data_dir, "SRP092402.tsv")
metadata_file <- file.path(data_dir, "metadata_SRP092402.tsv")
output_file <- file.path(data_dir, "SRP092402_Symbols.tsv")

# Read in metadata and expression data
metadata <- readr::read_tsv(metadata_file)
expression_df <- readr::read_tsv(data_file) %>%
  tibble::column_to_rownames("Gene")

# Make sure the sample columns are in the same order as the metadata
expression_df <- expression_df %>%
  dplyr::select(metadata$refinebio_accession_code)
stopifnot(all.equal(colnames(expression_df), metadata$refinebio_accession_code))

expression_df <- expression_df %>%
  tibble::rownames_to_column("Gene")

# Map Ensembl IDs to gene symbols, keeping every symbol an ID maps to
mapped_list <- mapIds(
  org.Hs.eg.db,
  keys = expression_df$Gene,
  keytype = "ENSEMBL",
  column = "SYMBOL",
  multiVals = "list"
)

mapped_df <- mapped_list %>%
  tibble::enframe(name = "Ensembl", value = "Symbol") %>%
  tidyr::unnest(cols = Symbol)

# Report how well the mapping went
message("Ensembl IDs with no symbol: ", sum(is.na(mapped_df$Symbol)))
multi_mapped <- mapped_df %>%
  dplyr::count(Ensembl, name = "n_symbols") %>%
  dplyr::filter(n_symbols > 1)
message("Ensembl IDs mapping to more than one symbol: ", nrow(multi_mapped))

# Collapse multiple symbols into one semicolon-separated column
collapsed_mapped_df <- mapped_df %>%
  dplyr::group_by(Ensembl) %>%
  dplyr::summarize(all_symbols = paste(Symbol, collapse = ";"))

# Keep the first mapped symbol as the main gene name, then join the expression data
final_mapped_df <- data.frame(
  "first_mapped_symbol" = mapIds(
    org.Hs.eg.db,
    keys = expression_df$Gene,
    keytype = "ENSEMBL",
    column = "SYMBOL",
    multiVals = "first"
  )
) %>%
  tibble::rownames_to_column("Ensembl") %>%
  dplyr::inner_join(collapsed_mapped_df, by = "Ensembl") %>%
  dplyr::inner_join(expression_df, by = c("Ensembl" = "Gene"))

readr::write_tsv(final_mapped_df, output_file)
message("Wrote ", output_file)
