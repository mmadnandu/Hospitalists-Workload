#---------------------------------------------------------------
# Project: Clinicians' workload
# Author: Md Mohiuddin Adnan
# Date: 07/15/2025
#---------------------------------------------------------------

# Load packages
library(tidyverse)
library(caret)
library(nnet)
library(randomForest)
library(xgboost)
library(e1071)
library(kknn)
library(binom)
library(pROC)
library(readxl)
library(janitor)
library(bit64)
options(scipen = 999)
#--------------------------------------------------------------------------------
# Function to calculate Accuracy, Precision, Recall, AUC with 95% CI
#--------------------------------------------------------------------------------

run_wl_prediction_analysis <- function(data, target_col, test_frac = 0.2, k_folds = 5) {
  
  set.seed(555)
  
  results_list <- list()
  
  #-----------------------------
  # 1. Stratified Test Split
  #-----------------------------
  train_idx <- createDataPartition(data[[target_col]], p = 1 - test_frac, list = FALSE)
  train_data <- data[train_idx, ] %>% droplevels()
  test_data  <- data[-train_idx, ] %>% droplevels()
  
  cat("\n✅ Stratified split done: ", nrow(train_data), "train /", nrow(test_data), "test\n")
  
  #-----------------------------
  # 2. Run models with CV on training set
  #-----------------------------
  run_models <- function(train_data, test_data, y_colname, group_label) {
    
    train_data[[y_colname]] <- as.factor(train_data[[y_colname]])
    test_data[[y_colname]]  <- factor(test_data[[y_colname]], levels = levels(train_data[[y_colname]]))
    
    folds <- createFolds(train_data[[y_colname]], k = k_folds, list = TRUE, returnTrain = TRUE)
    
    model_metrics <- list()
    
    eval_on_test <- function(model, X_test, y_test, method) {
      # Predict
      if (inherits(model, "glm")) {  # Binary logistic
        prob <- predict(model, newdata = X_test, type = "response")
        pred <- ifelse(prob > 0.5, levels(y_test)[2], levels(y_test)[1])
      } else if (inherits(model, "nnet")) {  # Multinomial
        prob <- predict(model, newdata = X_test, type = "probs")
        pred <- levels(y_test)[max.col(prob)]
      } else {  # fallback
        pred <- predict(model, newdata = X_test)
        prob <- NULL
      }
      
      pred <- factor(pred, levels = levels(y_test))
      
      cm <- confusionMatrix(pred, y_test)
      
      # Compute TP/TN/FP/FN
      if (length(levels(y_test)) == 2) {
        TP <- cm$table[2,2]; TN <- cm$table[1,1]
        FP <- cm$table[1,2]; FN <- cm$table[2,1]
      } else {
        TP <- TN <- FP <- FN <- numeric(length(levels(y_test)))
        names(TP) <- levels(y_test)
        for (cls in levels(y_test)) {
          y_true_bin <- ifelse(y_test == cls, 1, 0)
          y_pred_bin <- ifelse(pred == cls, 1, 0)
          TP[cls] <- sum(y_true_bin & y_pred_bin)
          TN[cls] <- sum(!y_true_bin & !y_pred_bin)
          FP[cls] <- sum(!y_true_bin & y_pred_bin)
          FN[cls] <- sum(y_true_bin & !y_pred_bin)
        }
        TP <- sum(TP); TN <- sum(TN); FP <- sum(FP); FN <- sum(FP)  # sum across classes
      }
      
      # Compute AUC if probabilities available
      auc <- NA
      if (!is.null(prob)) {
        if (length(levels(y_test)) == 2) {
          roc_obj <- tryCatch(roc(y_test, prob, levels = levels(y_test)), error = function(e) NULL)
          if (!is.null(roc_obj)) auc <- as.numeric(auc(roc_obj))
        } else if (is.matrix(prob)) {
          y_bin <- model.matrix(~ y_test - 1)
          aucs <- sapply(1:ncol(y_bin), function(k) {
            roc_obj <- tryCatch(roc(y_bin[,k], prob[,k], quiet = TRUE), error = function(e) NULL)
            if (!is.null(roc_obj)) as.numeric(auc(roc_obj)) else NA
          })
          auc <- mean(aucs, na.rm = TRUE)
        }
      }
      
      data.frame(
        Model = method,
        Accuracy = cm$overall["Accuracy"],
        Precision = if(length(cm$byClass) == 1) cm$byClass["Precision"] else mean(cm$byClass[,"Precision"], na.rm = TRUE),
        Recall = if(length(cm$byClass) == 1) cm$byClass["Recall"] else mean(cm$byClass[,"Recall"], na.rm = TRUE),
        AUC = auc,
        TP = TP, TN = TN, FP = FP, FN = FN
      )
    }
    
    #-----------------------------
    # Logistic / Multinomial
    #-----------------------------
    if (length(unique(train_data[[y_colname]])) > 2) {
      model <- multinom(as.formula(paste(y_colname, "~ .")), data = train_data, trace = FALSE)
      method_name <- "Multinomial Logistic"
    } else {
      model <- glm(as.formula(paste(y_colname, "~ .")), data = train_data, family = "binomial")
      method_name <- "Logistic Regression"
    }
    
    res_test <- eval_on_test(model, test_data %>% select(-all_of(y_colname)), test_data[[y_colname]], method_name)
    
    res_test$Group <- group_label
    results_list[[length(results_list)+1]] <<- res_test
  }
  
  #-----------------------------
  # Run analysis for original levels
  #-----------------------------
  original_levels <- sort(unique(as.numeric(as.character(data[[target_col]]))))
  n_levels <- length(original_levels)
  
  cat("\n====== 7-class Model ======\n")
  run_models(train_data, test_data, target_col, "7-class")
  
  cat("\n====== 3-class Groupings ======\n")
  for (i in 1:(n_levels - 2)) {
    for (j in (i + 1):(n_levels - 1)) {
      g1 <- original_levels[1:i]; g2 <- original_levels[(i+1):j]; g3 <- original_levels[(j+1):n_levels]
      grouped_data <- data %>% select(-all_of(target_col))
      grouped_data[[paste0(target_col, "_3")]] <- factor(
        ifelse(as.numeric(as.character(data[[target_col]])) %in% g1, "Low",
               ifelse(as.numeric(as.character(data[[target_col]])) %in% g2, "Medium", "High"))
      )
      run_models(train_data %>% select(-all_of(target_col)) %>% mutate(!!paste0(target_col,"_3") := grouped_data[[paste0(target_col,"_3")]]),
                 test_data %>% select(-all_of(target_col)) %>% mutate(!!paste0(target_col,"_3") := grouped_data[[paste0(target_col,"_3")]]),
                 paste0(target_col, "_3"),
                 sprintf("3-class (%s)/(%s)/(%s)", paste(g1, collapse=","), paste(g2, collapse=","), paste(g3, collapse=",")))
    }
  }
  
  cat("\n====== All 2-class Groupings ======\n")
  for (i in 1:(n_levels-1)) {
    g1 <- original_levels[1:i]; g2 <- original_levels[(i+1):n_levels]
    grouped_data <- data %>% select(-all_of(target_col))
    grouped_data[[paste0(target_col, "_2")]] <- factor(
      ifelse(as.numeric(as.character(data[[target_col]])) %in% g1, "Low", "High")
    )
    run_models(train_data %>% select(-all_of(target_col)) %>% mutate(!!paste0(target_col,"_2") := grouped_data[[paste0(target_col,"_2")]]),
               test_data %>% select(-all_of(target_col)) %>% mutate(!!paste0(target_col,"_2") := grouped_data[[paste0(target_col,"_2")]]),
               paste0(target_col, "_2"),
               sprintf("2-class (%s) vs (%s)", paste(g1, collapse=","), paste(g2, collapse=",")))
  }
  
  #-----------------------------
  # Save results
  #-----------------------------
  results_df <- bind_rows(results_list)
  dir.create("results", showWarnings = FALSE)
  write.csv(results_df,
            file = file.path("results", paste0("model_performance_results_", target_col, ".csv")),
            row.names = FALSE)
  message(paste0("✅ All results saved to 'results/model_performance_results_", target_col, ".csv'"))
  
  return(results_df)
}


#--------------------------------------------------------------------------------
# Run analysis for workload variables
#--------------------------------------------------------------------------------
final_data_for_analysis <- read_excel("data/final_data_for_analysis.xlsx") %>%
  janitor::clean_names() %>%
  mutate(wl_total = pmin(pmax(round(wl_total), 1), 7) %>% as.factor())

# target_vars <- "wl_physical"

# List of workload target variables
target_vars <- c("wl_physical", "wl_mental", "wl_temporal",
                 "wl_performance", "wl_effort", "wl_frustration", "wl_total"
)


feature_cols <- c(
  "length_of_patient_stay_hrs", "role", "days_patient_seen", "rvu_total", 
  "patient_contact_time", "complexity_type", "complexity", "chart_review_activities",
  "current_cls", "max_cls", "charlson_comorbidity_index_score_column",
  "ach_eligibility_v3_score", "workload_acuity_score"
)

for (var in target_vars) {
  cat("\n\n===========================\nRunning analysis for:", var, "\n===========================\n")
  data_for_model <- final_data_for_analysis %>% select(all_of(c(var, feature_cols))) %>% drop_na()
  data_for_model[[var]] <- as.factor(data_for_model[[var]])
  run_wl_prediction_analysis(data_for_model, var)
}


#########################################
# 02/20/2026
#########################################
Length_of_Patient_Stay_hrs,Days_Patient_Seen,Patient_Contact_mins,RVU_Total,patient_contact_time,