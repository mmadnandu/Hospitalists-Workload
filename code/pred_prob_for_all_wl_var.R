#--------------------------------------------------------------------------------
# Run analysis for workload variables
#--------------------------------------------------------------------------------
library(readxl)
library(dplyr)
library(janitor)
library(nnet)      
library(openxlsx)    
library(readxl)
library(tidyverse)


set.seed(3601)

# Read and prepare the data
final_data_for_analysis <- read_tsv("data/ModelingData.txt") %>%
  janitor::clean_names() 

# List of workload target variables
target_vars <- c(
  "wl_physical", "wl_mental", "wl_temporal",
  "wl_performance", "wl_effort", "wl_frustration"
)

# List of feature columns
feature_cols <- c(
  "los","role", "dps", "rvu", "time",
  "complexity", "chart",
  "ccls", "mcls",
  "cci",
  "ach",
  "was"
)

# Create a new Excel workbook to store all results
wb <- createWorkbook()

# var <- "wl_physical"
# Loop through each workload variable
for (var in target_vars) {
  cat("\n\n===========================\nRunning analysis for:", var, "\n===========================\n")
  
  # Prepare data for modeling
  data_for_model <- final_data_for_analysis %>%
    select(all_of(c(var, feature_cols))) %>%
    drop_na()
  
  data_for_model[[var]] <- as.factor(data_for_model[[var]])
  
  # Fit multinomial logistic regression
  formula_str <- as.formula(paste(var, "~ ."))
  model <- multinom(formula_str, data = data_for_model, trace = FALSE)
  
  # Predicted probabilities for each class
  predicted_probs <- as.data.frame(predict(model, type = "probs"))
  # Get the predicted class (highest probability)
  predicted_label <- colnames(predicted_probs)[max.col(predicted_probs)]
  predicted_prob <- apply(predicted_probs, 1, max)
  
  # Combine only target var + predicted label + highest probability
  results <- data_for_model %>%
    select(all_of(var)) %>%
    mutate(
      predicted_label = predicted_label,
      predicted_probability = predicted_prob
    )
  
  # Add to workbook
  addWorksheet(wb, var)
  writeData(wb, var, results)
}

# Save the Excel file
output_path <- "results/multinomial_logistic_predictions.xlsx"
saveWorkbook(wb, output_path, overwrite = TRUE)
cat("\n✅ All results saved to:", output_path, "\n")
