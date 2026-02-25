library(officer)
library(flextable)
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

#---------------------------------------------------------------
# 2. Prepare datasets
#---------------------------------------------------------------

dep_data_2 <- PCA_data_2 %>% select(all_of(dependent_vars_2))

#---------------------------------------------------------------
# 3. PCA on dependent variables (workload outcomes)
#---------------------------------------------------------------

# PCA with scaling
pca_dep <- prcomp(dep_data_2, scale. = TRUE)
summary(pca_dep)
pca_dep$rotation
plot(pca_dep, type = "l", main = "Scree Plot - Dependent Variables")

#---------------------------------------------------------------
# 5. Get PCA scores if you want to use them in modeling
#---------------------------------------------------------------
dep_scores <- as.data.frame(pca_dep$x)   

#-----------------------------------------
# plot for all dependent variables
#-----------------------------------------
# Combine dep_scores and all columns from dep_data_2
dep_scores_all <- bind_cols(dep_scores, dep_data_2)
write.csv(dep_scores_all, "data/dep_scores_all.csv", row.names = F)
# List of dependent columns
dep_cols <- colnames(dep_data_2)


ggplot(dep_scores_all, aes(x = PC1, fill = factor(frustration))) + # mental physical performance effort frustration
  geom_density(alpha = 0.4) +
  labs(
    title = "Distribution of PC1 by Physical Class",
    x = "PC1",
    y = "Density",
    fill = "Physical Class"
  ) +
  theme_minimal(base_size = 14)

#------------------------------------------------------------
pca_res <- prcomp(dep_data_2[, dependent_vars_2], scale. = TRUE)
summary(pca_res)

dep_scores_all <- as.data.frame(pca_res$x[, 1:3])

# Get explained variance ratio for weighting
var_explained <- summary(pca_res)$importance["Proportion of Variance", 1:3]

#-----------------------------
# split each PCs into classes
#-----------------------------

# two class
for (pc in c("PC1", "PC2", "PC3")) {
  
  dep_scores_all[[paste0(pc, "_class2")]] <- ifelse(
    dep_scores_all[[pc]] <= median(dep_scores_all[[pc]], na.rm = TRUE),
    "Low", 
    "High"
  )
  
  # Make it a factor (optional)
  dep_scores_all[[paste0(pc, "_class2")]] <- factor(
    dep_scores_all[[paste0(pc, "_class2")]],
    levels = c("Low", "High")
  )
}


# three classes
for (pc in c("PC1", "PC2", "PC3")) {
  dep_scores_all[[paste0(pc, "_class3")]] <- cut(
    dep_scores_all[[pc]],
    breaks = quantile(dep_scores_all[[pc]], probs = c(0, 1/3, 2/3, 1), na.rm = TRUE),
    labels = c("Low", "Medium", "High"),
    include.lowest = TRUE
  )
}


# Weighted composite workload score
dep_scores_all$Total_score_PCA <- as.numeric(as.matrix(dep_scores_all[, 1:3]) %*% var_explained)

dep_scores_all$workload_class <- cut(
  dep_scores_all$Total_score_PCA,
  breaks = quantile(dep_scores_all$Total_score_PCA, probs = c(0, 1/3, 2/3, 1), na.rm = TRUE),
  labels = c("Low", "Medium", "High"),
  include.lowest = TRUE
)

# Merge everything
PCA_final_score_data <- dep_data_2 %>%
  mutate(
    PC1             = dep_scores_all$PC1,
    PC2             = dep_scores_all$PC2,
    PC3             = dep_scores_all$PC3,
    
    # Binary Low/High classes
    PC1_class2      = dep_scores_all$PC1_class2,
    PC2_class2      = dep_scores_all$PC2_class2,
    PC3_class2      = dep_scores_all$PC3_class2,
    
    # Low/Medium/High classes
    PC1_class3      = dep_scores_all$PC1_class3,
    PC2_class3      = dep_scores_all$PC2_class3,
    PC3_class3      = dep_scores_all$PC3_class3,
    
    # Total score class
    workload_class  = dep_scores_all$workload_class
  )


write.csv(PCA_final_score_data, "data/PCA_final_score_data.csv",row.names = F)

#---------------------------------------------------------------------------------
# compare workload variable classes and PCs classes
#---------------------------------------------------------------------------------
PCA_final_score_data <- read_csv("data/PCA_final_score_data.csv")
df <- PCA_final_score_data

# Variables to categorize
vars <- c("physical", "mental", "temporal", "performance", "effort", "frustration")

# ---- FUNCTION FOR 2-CLASS SPLIT USING MEDIAN WITH TIE HANDLING ----
categorize_2class <- function(x) {
  med <- median(x, na.rm = TRUE)
  
  # Check distribution around the median
  lower <- sum(x < med)
  equal <- sum(x == med)
  upper <- sum(x > med)
  
  # If many equals at median, split them to balance groups
  if (equal > 0) {
    # Rank median values randomly but reproducibly
    set.seed(123)
    ranks <- rank(runif(equal))
    
    # Number to put in Low is whichever group is smaller
    need_low <- max(0, ceiling((length(x)/2) - lower))
    
    new_med_low <- ranks <= need_low
    
    out <- ifelse(
      x < med, "Low",
      ifelse(x > med, "High",
             ifelse(new_med_low, "Low", "High"))
    )
  } else {
    out <- ifelse(x <= med, "Low", "High")
  }
  
  factor(out, levels = c("Low", "High"))
}

# ---- FUNCTION FOR 3-CLASS SPLIT USING 33%/66% QUANTILES WITH TIE HANDLING ----
categorize_3class <- function(x) {
  q <- quantile(x, probs = c(.33, .66), na.rm = TRUE)
  q1 <- q[1]
  q2 <- q[2]
  
  # Handle ties at boundaries
  # Random distribution of boundary ties for more balanced classes
  set.seed(123)
  
  x_class <- character(length(x))
  
  # 1. Values below q1
  x_class[x < q1] <- "Low"
  
  # 2. Values above q2
  x_class[x > q2] <- "High"
  
  # 3. Values equal to q1 or q2 → split randomly but reproducibly
  eq_q1 <- which(x == q1)
  eq_q2 <- which(x == q2)
  
  if (length(eq_q1) > 0) {
    n_low <- max(0, round(length(x) * 0.33) - sum(x < q1))
    low_idx <- sample(eq_q1, min(n_low, length(eq_q1)))
    x_class[eq_q1] <- ifelse(seq_along(eq_q1) %in% low_idx, "Low", "Medium")
  }
  
  if (length(eq_q2) > 0) {
    n_high <- max(0, round(length(x) * 0.33) - sum(x > q2))
    high_idx <- sample(eq_q2, min(n_high, length(eq_q2)))
    x_class[eq_q2] <- ifelse(seq_along(eq_q2) %in% high_idx, "High", "Medium")
  }
  
  # 4. Remaining: between q1 and q2
  x_class[x > q1 & x < q2] <- "Medium"
  
  factor(x_class, levels = c("Low", "Medium", "High"))
}

# ---- APPLY THE FUNCTIONS TO THE VARIABLES ----
df_cat <- df %>%
  mutate(across(all_of(vars), categorize_2class, .names = "{.col}_class2_raw")) %>%
  mutate(across(all_of(vars), categorize_3class, .names = "{.col}_class3_raw"))

write.csv(df_cat, "data/PCA_workload_classes_data.csv",row.names = F)

# ---- Optionally compare with corresponding PC classes ----
# Example: confusion table for physical vs PC1
table(df_cat$mental_class2_raw, df_cat$PC1_class2)
table(df_cat$physical_class3_raw, df_cat$PC1_class3)

# workload variables
vars <- c("physical", "mental", "temporal", "performance", "effort", "frustration")
# Define level orders
levels2 <- c("Low", "High")
levels3 <- c("Low", "Medium", "High")
pretty_label <- function(x) gsub("_", " ", tools::toTitleCase(x))

# PC indices (only 3 PCs exist)
pcs <- 1:3

# Capture all printed output into a character vector
output_text <- capture.output({
  
  for (v in vars) {
    
    cat("\n================ ", toupper(v), " ================\n")
    
    ## CLASS 2 TABLES
    raw2 <- paste0(v, "_class2_raw")
    df_cat[[raw2]] <- factor(df_cat[[raw2]], levels = levels2)
    
    for (pc in pcs) {
      pc2 <- paste0("PC", pc, "_class2")
      df_cat[[pc2]] <- factor(df_cat[[pc2]], levels = levels2)
      
      cat("\n--- ", pretty_label(raw2), " vs ", pretty_label(pc2), " ---\n", sep = "")
      
      tbl2 <- table(df_cat[[raw2]], df_cat[[pc2]])
      dimnames(tbl2) <- list(Raw = levels2, PC = levels2)
      print(tbl2)
    }
    
    ## CLASS 3 TABLES
    raw3 <- paste0(v, "_class3_raw")
    df_cat[[raw3]] <- factor(df_cat[[raw3]], levels = levels3)
    
    for (pc in pcs) {
      pc3 <- paste0("PC", pc, "_class3")
      df_cat[[pc3]] <- factor(df_cat[[pc3]], levels = levels3)
      
      cat("\n--- ", pretty_label(raw3), " vs ", pretty_label(pc3), " ---\n", sep = "")
      
      tbl3 <- table(df_cat[[raw3]], df_cat[[pc3]])
      dimnames(tbl3) <- list(Raw = levels3, PC = levels3)
      print(tbl3)
    }
  }
  
  cat("\n================ Rotation (Correlation/Contribution) Matrix ================\n")
  pca_res <- prcomp(dep_data_2[, dependent_vars_2], scale. = TRUE)
  print(pca_res)
})


doc <- read_docx()

doc <- body_add_par(doc, "PCA results and contingency tables", style = "heading 1")

# add as preformatted text
doc <- body_add_par(doc, paste(output_text, collapse = "\n"), style = "Normal")

print(doc, target = "PCA_output.docx")


sink("pca_output.txt")   # Start writing output to file

for (v in vars) {
  
  cat("\n================ ", toupper(v), " ================\n")
  
  ## -------------------------------
  ## FIRST print all CLASS 2 tables
  ## -------------------------------
  raw2 <- paste0(v, "_class2_raw")
  df_cat[[raw2]] <- factor(df_cat[[raw2]], levels = levels2)
  
  for (pc in pcs) {
    
    pc2 <- paste0("PC", pc, "_class2")
    df_cat[[pc2]] <- factor(df_cat[[pc2]], levels = levels2)
    
    cat("\n--- ", pretty_label(raw2), " vs ", pretty_label(pc2), " ---\n", sep = "")
    
    tbl2 <- table(df_cat[[raw2]], df_cat[[pc2]])
    dimnames(tbl2) <- list(Raw = levels2, PC = levels2)
    print(tbl2)
  }
  
  ## -------------------------------
  ## NEXT print all CLASS 3 tables
  ## -------------------------------
  raw3 <- paste0(v, "_class3_raw")
  df_cat[[raw3]] <- factor(df_cat[[raw3]], levels = levels3)
  
  for (pc in pcs) {
    
    pc3 <- paste0("PC", pc, "_class3")
    df_cat[[pc3]] <- factor(df_cat[[pc3]], levels = levels3)
    
    cat("\n--- ", pretty_label(raw3), " vs ", pretty_label(pc3), " ---\n", sep = "")
    
    tbl3 <- table(df_cat[[raw3]], df_cat[[pc3]])
    dimnames(tbl3) <- list(Raw = levels3, PC = levels3)
    print(tbl3)
  }
}

cat("\n================ Rotation (Correlation/Contribution) Matrix ================\n")
print(pca_res)

sink()   # Stop writing and close file








for (v in vars) {
  
  cat("\n================ ", toupper(v), " ================\n")
  
  ## -------------------------------
  ## FIRST print all CLASS 2 tables
  ## -------------------------------
  raw2 <- paste0(v, "_class2_raw")
  df_cat[[raw2]] <- factor(df_cat[[raw2]], levels = levels2)
  
  for (pc in pcs) {
    
    pc2 <- paste0("PC", pc, "_class2")
    df_cat[[pc2]] <- factor(df_cat[[pc2]], levels = levels2)
    
    cat("\n--- ", pretty_label(raw2), " vs ", pretty_label(pc2), " ---\n", sep = "")
    
    tbl2 <- table(df_cat[[raw2]], df_cat[[pc2]])
    
    dimnames(tbl2) <- list(
      Raw = levels2,
      PC  = levels2
    )
    
    print(tbl2)
  }
  
  
  ## -------------------------------
  ## NEXT print all CLASS 3 tables
  ## -------------------------------
  raw3 <- paste0(v, "_class3_raw")
  df_cat[[raw3]] <- factor(df_cat[[raw3]], levels = levels3)
  
  for (pc in pcs) {
    
    pc3 <- paste0("PC", pc, "_class3")
    df_cat[[pc3]] <- factor(df_cat[[pc3]], levels = levels3)
    
    cat("\n--- ", pretty_label(raw3), " vs ", pretty_label(pc3), " ---\n", sep = "")
    
    tbl3 <- table(df_cat[[raw3]], df_cat[[pc3]])
    
    dimnames(tbl3) <- list(
      Raw = levels3,
      PC  = levels3
    )
    
    print(tbl3)
  }
}

cat("\n================ Rotation (Correlation/Contribution) Matrix ================\n")
pca_res
