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

