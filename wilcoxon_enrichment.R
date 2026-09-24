
library(AnnotationDbi)   
library(org.Hs.eg.db)   
library(GO.db)         

min_term_size <- 10
fdr_cutoff    <- 0.05
auc_gap       <- 0.1 

de <- read.delim("results/diff_expr_tb_vs_healthy.tsv")

scores <- de$t_statistic
names(scores) <- de$Ensembl

sig_genes <- de$Ensembl[de$direction != "Not significant"]

print(paste("Genes tested:", length(scores)))
print(paste("Significant genes:", length(sig_genes)))

id_table <- AnnotationDbi::select(org.Hs.eg.db, keys = de$Ensembl,
                                  keytype = "ENSEMBL", columns = "ENTREZID")
id_table <- id_table[!is.na(id_table$ENTREZID), ]

go_lists <- as.list(org.Hs.egGO2ALLEGS)
all_go_ids <- names(go_lists)

go_category <- Ontology(all_go_ids)
bp_ids <- all_go_ids[go_category %in% "BP"]
print(paste("GO Biological Process terms to check:", length(bp_ids)))


go_id_col   <- c()
size_col    <- c()
n_sig_col   <- c()
auc_col     <- c()
pvalue_col  <- c()

for (go_id in bp_ids) {
  
  entrez_in_term <- unique(unlist(go_lists[[go_id]]))
  
  genes_in_term <- unique(id_table$ENSEMBL[id_table$ENTREZID %in% entrez_in_term])
  
  if (length(genes_in_term) < min_term_size) {
    next
  }
  
  scores_in  <- scores[genes_in_term]
  scores_out <- scores[!(names(scores) %in% genes_in_term)]
  
  test <- wilcox.test(scores_in, scores_out, exact = FALSE)
  
  auc <- as.numeric(test$statistic) / (length(scores_in) * length(scores_out))
  
  go_id_col  <- c(go_id_col,  go_id)
  size_col   <- c(size_col,   length(genes_in_term))
  n_sig_col  <- c(n_sig_col,  sum(genes_in_term %in% sig_genes))
  auc_col    <- c(auc_col,    auc)
  pvalue_col <- c(pvalue_col, test$p.value)
}

results <- data.frame(
  GO_ID             = go_id_col,
  term              = as.character(Term(GOTERM[go_id_col])),
  annotated         = size_col, 
  significant_genes = n_sig_col, 
  auc               = auc_col,
  pvalue_wilcoxon   = pvalue_col
)

results$padj_wilcoxon_BH <- p.adjust(results$pvalue_wilcoxon, method = "BH")

results$direction <- ifelse(results$auc > 0.5, "Higher in TB", "Lower in TB")

results$significant <- results$padj_wilcoxon_BH < fdr_cutoff &
  abs(results$auc - 0.5) >= auc_gap

results <- results[order(results$pvalue_wilcoxon), ]


write.table(results, "results/wilcoxon_GO_BP_tb_vs_healthy.tsv",
            sep = "\t", quote = FALSE, row.names = FALSE)

print(paste("GO terms tested:", nrow(results)))
print(paste("Significant terms:", sum(results$significant)))
print(paste("  higher in TB:", sum(results$significant & results$direction == "Higher in TB")))
print(paste("  lower in TB: ", sum(results$significant & results$direction == "Lower in TB")))
print("Saved: results/wilcoxon_GO_BP_tb_vs_healthy.tsv")