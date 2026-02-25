# ---------------------------------------------------------------
# Project: Workload 
# Author: Md Mohiuddin Adnan
# Date: 2025-08-13
# ---------------------------------------------------------------

# libraries
library(psych)       # For PCA, parallel analysis
library(GPArotation) # For rotation options
library(janitor)     
library(readxl)
library(tidyverse)
library(ggplot2)
library(plotly)

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

#---------------------------------------------------------------
# 4. PCA on dependent variables (workload outcomes)
#---------------------------------------------------------------

# PCA with scaling
pca_dep <- prcomp(dep_data_2, scale. = TRUE)
summary(pca_dep)
pca_dep$rotation
plot(pca_dep, type = "l", main = "Scree Plot - Dependent Variables")


#---------------------------------------------------------------
# 5. Get PCA scores if you want to use them in modeling
#---------------------------------------------------------------
dep_scores <- as.data.frame(pca_dep$x)   # PC1, PC2, ...


#---------------------------------------------------------------
# 6. scatter plots of PCs
#---------------------------------------------------------------
dep_scores_physical <- dep_scores %>%
  mutate(physical = dep_data_2$physical)

p <- ggplot(dep_scores_physical, aes(x = PC1, y = PC2, color = factor(physical),
                                     text = paste("PC1:", PC1,
                                                  "<br>PC2:", PC2,
                                                  "<br>Physical:", physical))) +
  geom_point(alpha = 0.7, size = 1) +
  theme_minimal() +
  labs(
    title = "PC1 vs PC2 Scatter Plot",
    x = "Principal Component 1",
    y = "Principal Component 2",
    color = "Physical"
  )

interactive_plot <- ggplotly(p, tooltip = "text")
interactive_plot
htmlwidgets::saveWidget(interactive_plot, "reports/PC_scatter_plot.html")

#-----------------------------------------
# 7. plot for all dependent variables
#-----------------------------------------

# Combine dep_scores and all columns from dep_data_2
dep_scores_all <- bind_cols(dep_scores, dep_data_2)
write.csv(dep_scores_all, "data/dep_scores_all.csv", row.names = F)
# List of dependent columns
dep_cols <- colnames(dep_data_2)

# Loop over each dependent column to create and save a plot
for(dep_var in dep_cols) {
  
  # Create ggplot
  p <- ggplot(dep_scores_all, 
              aes(x = PC1, y = PC2,
                  color = factor(.data[[dep_var]]),  # dynamic column
                  text = paste("PC1:", PC1,
                               "<br>PC2:", PC2,
                               "<br>", dep_var, ":", .data[[dep_var]]))) +
    geom_point(alpha = 0.7, size = 1.5) +
    theme_minimal() +
    labs(
      title = paste("PC1 vs PC2 Scatter Plot", dep_var),
      x = "PC1",
      y = "PC2",
      color = dep_var
    )
  
  # Convert to interactive Plotly
  interactive_plot <- ggplotly(p, tooltip = "text")
  
  # Save as HTML
  htmlwidgets::saveWidget(interactive_plot, paste0("reports/",dep_var, "_PC1_PC2_scatter.html"))
}



#------------------------------------------------
# 8. PCAs for all all combination of likert scale
#------------------------------------------------

discretize_var <- function(x, breaks) {
  # x = numeric vector
  # breaks = vector of numeric values defining split points
  cut(x, breaks = c(-Inf, breaks, Inf), labels = FALSE)
}

two_class_splits <- lapply(1:6, function(i) i)  # thresholds 1:6

three_class_splits <- list()
for (i in 1:5) {       # lower split point
  for (j in (i+1):6) { # upper split point
    three_class_splits <- append(three_class_splits, list(c(i,j)))
  }
}

pca_results <- list()

# 2-class
for (threshold in two_class_splits) {
  dep_cat <- dep_data_2 %>% mutate(across(everything(), ~discretize_var(.x, breaks = threshold)))
  pca <- prcomp(dep_cat, scale. = TRUE)
  pca_results[[paste0("2class_", threshold)]] <- pca
}

# 3-class
for (thresholds in three_class_splits) {
  dep_cat <- dep_data_2 %>% mutate(across(everything(), ~discretize_var(.x, breaks = thresholds)))
  pca <- prcomp(dep_cat, scale. = TRUE)
  pca_results[[paste0("3class_", paste(thresholds, collapse="_"))]] <- pca
}



#######################
# Initialize lists to store results
all_loadings <- list()
all_variance <- list()

# Function to perform PCA and save loadings + variance
save_pca_results <- function(dep_matrix, split_type, combination_name, id_name) {
  pca <- prcomp(dep_matrix, scale. = TRUE)
  pca_sum <- summary(pca)
  
  # Loadings
  loadings <- as.data.frame(pca$rotation)
  loadings$variable <- rownames(loadings)
  loadings$split_type <- split_type
  loadings$combination <- combination_name
  all_loadings[[id_name]] <<- loadings
  
  # Variance explained
  var_df <- as.data.frame(pca_sum$importance)
  var_df$measure <- rownames(var_df)
  var_df$split_type <- split_type
  var_df$combination <- combination_name
  all_variance[[id_name]] <<- var_df
}

# ------------------------
# 1. Original 7-class PCA
save_pca_results(
  dep_data_2,
  split_type = "7-class",
  combination_name = "original_1-7",
  id_name = "7class_original"
)

# ------------------------
# 2. 2-class PCA
for (threshold in two_class_splits) {
  dep_cat <- dep_data_2 %>% mutate(across(everything(), ~discretize_var(.x, breaks = threshold)))
  save_pca_results(dep_cat,
                   split_type = "2-class",
                   combination_name = paste0("1-", threshold, " vs ", threshold+1, "-7"),
                   id_name = paste0("2class_", threshold))
}

# ------------------------
# 3. 3-class PCA
for (thresholds in three_class_splits) {
  dep_cat <- dep_data_2 %>% mutate(across(everything(), ~discretize_var(.x, breaks = thresholds)))
  save_pca_results(dep_cat,
                   split_type = "3-class",
                   combination_name = paste0("1-", thresholds[1], " / ",
                                             thresholds[1]+1, "-", thresholds[2], " / ",
                                             thresholds[2]+1, "-7"),
                   id_name = paste0("3class_", paste(thresholds, collapse="_")))
}

# ------------------------
# Combine and save
loadings_all_df <- bind_rows(all_loadings, .id = "run_id") %>%
  select(variable, split_type, combination, everything(), -run_id)

variance_all_df <- bind_rows(all_variance, .id = "run_id") %>%
  select(measure, split_type, combination, everything(), -run_id)

# Save to CSV
write.csv(loadings_all_df, "results/PCA_loadings_all_combinations.csv", row.names = FALSE)
write.csv(variance_all_df, "results/PCA_variance_all_combinations.csv", row.names = FALSE)



#---------------------------------------------------------------
# 7-likert scales' classes  distribution 
#---------------------------------------------------------------

data <- dep_data_2
target_cols <- names(data)

results <- list()

for (target_col in target_cols) {
  cat("\n==============================\n")
  cat("Target:", target_col, "\n")
  cat("==============================\n")
  
  original_levels <- sort(unique(data[[target_col]]))
  n_levels <- length(original_levels)
  
  # --- 7-class distribution ---
  dist_7 <- as.data.frame(table(class = data[[target_col]])) %>%
    mutate(
      percent = round(100 * Freq / sum(Freq), 2),
      target = target_col,
      grouping_type = "7-class",
      grouping_desc = paste(original_levels, collapse = ",")
    ) %>%
    select(target, grouping_type, grouping_desc, class, count = Freq, percent)
  
  results[[length(results) + 1]] <- dist_7
  
  # --- 3-class groupings ---
  for (i in 1:(n_levels - 2)) {
    for (j in (i + 1):(n_levels - 1)) {
      g1 <- original_levels[1:i]
      g2 <- original_levels[(i + 1):j]
      g3 <- original_levels[(j + 1):n_levels]
      
      grouped_data <- factor(
        ifelse(data[[target_col]] %in% g1, "Low",
               ifelse(data[[target_col]] %in% g2, "Medium", "High")),
        levels = c("Low", "Medium", "High")
      )
      
      dist_3 <- as.data.frame(table(class = grouped_data)) %>%
        mutate(
          percent = round(100 * Freq / sum(Freq), 2),
          target = target_col,
          grouping_type = "3-class",
          grouping_desc = sprintf("(%s)/(%s)/(%s)",
                                  paste(g1, collapse = ","),
                                  paste(g2, collapse = ","),
                                  paste(g3, collapse = ","))
        ) %>%
        select(target, grouping_type, grouping_desc, count = Freq, percent, class)
      
      results[[length(results) + 1]] <- dist_3
    }
  }
  
  # --- 2-class groupings ---
  for (i in 1:(n_levels - 1)) {
    g1 <- original_levels[1:i]
    g2 <- original_levels[(i + 1):n_levels]
    
    grouped_data <- factor(
      ifelse(data[[target_col]] %in% g1, "Low", "High"),
      levels = c("Low", "High")
    )
    
    dist_2 <- as.data.frame(table(class = grouped_data)) %>%
      mutate(
        percent = round(100 * Freq / sum(Freq), 2),
        target = target_col,
        grouping_type = "2-class",
        grouping_desc = sprintf("(%s) vs (%s)",
                                paste(g1, collapse = ","),
                                paste(g2, collapse = ","))
      ) %>%
      select(target, grouping_type, grouping_desc, count = Freq, percent, class)
    
    results[[length(results) + 1]] <- dist_2
  }
}

# --- Combine all results ---
final_results <- bind_rows(results)

# --- Save to CSV ---
write.csv(final_results, "data/class_distributions.csv", row.names = FALSE)



# #---------------------------------------------------------------
# # 8. Example: Model first dep PC using first 2 ind PCs
# #---------------------------------------------------------------
# keep first N PCs (example: first 2 PCs from each set)
# dep_scores_reduced <- dep_scores %>% select(PC1, PC2)
# lm_model <- lm(PC1 ~ PC1 + PC2,
#                data = cbind(dep_scores_reduced, ind_scores_reduced))
# summary(lm_model)
