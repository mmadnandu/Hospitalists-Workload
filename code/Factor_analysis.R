# =============================================
# Workload Factor Analysis & Score Computation
# =============================================

# libraries
library(psych)       # For factor analysis
library(GPArotation) # For rotation options
library(janitor)
library(readxl)
library(tidyverse)

#---------------------------------------------------------------
# 1. load dataset
#---------------------------------------------------------------

# Tab-delimited
PCA_data_2 <- read_tsv("data/ModelingData.txt")

# clean column names
PCA_data_2 <- janitor::clean_names(PCA_data_2)

dependent_vars_2 <- c(
  "physical", "mental", "temporal",
  "performance", "effort", "frustration"
)

independent_vars <- c(
  "los","role", "dps", "rvu", "time",
  "complexity", "chart",
  "ccls", "mcls",
  "cci",
  "ach",
  "was"
)

#---------------------------------------------------------------
# 2. Prepare datasets
#---------------------------------------------------------------

dep_data_2 <- PCA_data_2 %>% select(all_of(dependent_vars_2))

# -----------------------------
# 3. Check correlation matrix
# -----------------------------
cor_matrix <- cor(dep_data_2, use = "pairwise.complete.obs")
print(cor_matrix)


# -----------------------------
# 2. Parallel analysis to decide number of factors
# -----------------------------
fa.parallel(dep_data_2, fa = "fa")  # fa="fa" for factor analysis

# -----------------------------
# 3. Standardize the data
# -----------------------------
# Ensures variables on different scales are comparable
dep_data_scaled <- scale(dep_data_2)

# -----------------------------
# 4. Run factor analysis
# -----------------------------
# Choose number of factors suggested by parallel analysis (here nfactors=4)
fa_result <- fa(dep_data_scaled, nfactors = 4, rotate = "varimax", fm = "ml")

# -----------------------------
# 5. Inspect factor loadings
# -----------------------------
print(fa_result)  # hides small loadings < 0.3
fa_result$loadings          # full loadings

# -----------------------------
# 6. Compute factor scores for each patient
# -----------------------------
factor_scores <- factor.scores(dep_data_scaled, fa_result)$scores
factor_scores <- as.data.frame(factor_scores)
head(factor_scores)

# -----------------------------
# 7. Compute a single workload score
# -----------------------------
# Option A: simple sum of factor scores
workload_score <- rowSums(factor_scores)

# Option B: weighted by variance explained (SS loadings / sum)
weights <- fa_result$Vaccounted["SS loadings", ] / sum(fa_result$Vaccounted["SS loadings", ])
workload_score_weighted <- as.matrix(factor_scores) %*% weights

# Optional: make all positive (useful for allocation)
workload_score_pos <- workload_score - min(workload_score)

# Inspect final workload scores
head(workload_score_pos)

# -----------------------------
# 8. Next steps
# -----------------------------
# Use workload_score_pos to assign patients to physicians:
# - Sum current workload per physician
# - Assign new patients to the physician with least total workload
# - Repeat iteratively for balancing
