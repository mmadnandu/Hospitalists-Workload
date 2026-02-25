
#---------------------------------------------------------------
# Project: Clinicians' workload
# Author: Md Mohiuddin Adnan
# Date: 07/15/2025
#---------------------------------------------------------------


# packages
library(tidyverse)
library(caret)
library(nnet)
library(randomForest)
library(xgboost)
library(e1071)
library(class)
library(pROC)
library(readxl)
library(janitor)


#--------------------------------------------------------------------------------
# function to calcualte Accuracy, precision, recall, AUC for different work load
#--------------------------------------------------------------------------------

run_wl_prediction_analysis <- function(data, target_col) {

  # list for storing results
  results_list <- list()
  # k-fold
  k_folds <- 10
  
  #-----------------------------------------------------
  # run_model function for different ML methods
  #-----------------------------------------------------
  
  run_models <- function(data, y_colname, group_label) {
    
    set.seed(555)
    data[[y_colname]] <- as.factor(data[[y_colname]]) # y_colname: work load variable
    data <- data[!is.na(data[[y_colname]]), , drop = FALSE] # drop NAs
    folds <- createFolds(data[[y_colname]], k = k_folds, list = TRUE, returnTrain = FALSE)
    
    model_metrics <- list()
    
    #-------------------------------
    # k-fold loop
    #-------------------------------
    
    for (fold in 1:k_folds) {
      test_idx <- folds[[fold]]
      train <- data[-test_idx, ] %>% droplevels()
      test  <- data[test_idx, ] %>% droplevels()
      
      X_train <- train %>% select(-all_of(y_colname))
      X_test  <- test %>% select(-all_of(y_colname))
      
      # Ensure all factor columns in test match the levels in train
      for (col in names(X_train)) {
        if (is.factor(X_train[[col]])) {
          X_train[[col]] <- factor(X_train[[col]])  # drop unused levels
          X_test[[col]] <- factor(X_test[[col]], levels = levels(X_train[[col]]))
        }
      }
      
      # Align response variable factor levels
      y_train <- factor(train[[y_colname]])
      y_test  <- factor(test[[y_colname]], levels = levels(y_train))
      
      #-------------------------------
      # performance metrices
      #-------------------------------
      
      eval_model <- function(pred, prob = NULL, method = "", y_test, fold_index) {
        cm <- confusionMatrix(pred, y_test)
        acc <- cm$overall["Accuracy"]
        prec <- cm$byClass["Precision"]
        rec <- cm$byClass["Recall"]
        auc <- if (!is.null(prob) && length(levels(y_test)) == 2) {
          tryCatch({
            roc_obj <- roc(y_test, prob, levels = levels(y_test))
            as.numeric(auc(roc_obj))
          }, error = function(e) NA)
        } else { NA }
        
        model_metrics[[method]][[fold_index]] <<- data.frame(
          Accuracy = acc,
          Precision = prec,
          Recall = rec,
          AUC = auc
        )
      }
      
      #-------------------------------
      # Logistic / Multinomial
      #-------------------------------
      
      tryCatch({
        if (length(unique(y_train)) > 2) {
          model <- multinom(as.formula(paste(y_colname, "~ .")), data = train, trace = FALSE)
          pred <- predict(model, newdata = X_test)
          eval_model(as.factor(pred), NULL, "Multinomial Logistic", y_test, fold)
        } else {
          model <- glm(y_train ~ ., data = cbind(X_train, y_train), family = "binomial")
          prob <- predict(model, newdata = X_test, type = "response")
          pred <- ifelse(prob > 0.5, levels(y_train)[2], levels(y_train)[1])
          eval_model(as.factor(pred), prob, "Logistic Regression", y_test, fold)
        }
      }, error = function(e) {
        message("Logistic Error: ", e$message)
      })
      
      #-------------------------------
      # Random Forest
      #-------------------------------
      
      tryCatch({
        model <- randomForest(as.formula(paste(y_colname, "~ .")), data = train)
        pred <- predict(model, newdata = X_test)
        prob <- if (length(unique(y_train)) == 2) predict(model, newdata = X_test, type = "prob")[, 2] else NULL
        eval_model(as.factor(pred), prob, "Random Forest", y_test, fold)
      }, error = function(e) {
        message("Random Forest Error: ", e$message)
      })
      
      #-------------------------------
      # KNN
      #-------------------------------
      
      tryCatch({
        X_mat <- model.matrix(~ . - 1, data = rbind(X_train, X_test))
        X_scaled <- scale(X_mat)
        train_scaled <- X_scaled[1:nrow(X_train), ]
        test_scaled <- X_scaled[(nrow(X_train) + 1):nrow(X_scaled), ]
        pred <- knn(train_scaled, test_scaled, cl = y_train, k = 5)
        eval_model(as.factor(pred), NULL, "KNN", y_test, fold)
      }, error = function(e) {
        message("KNN Error: ", e$message)
      })
    }
    
    #-------------------------------
    # SVM
    #-------------------------------
    
    tryCatch({
      model <- svm(as.formula(paste(y_colname, "~ .")), data = train, probability = (length(unique(y_train)) == 2))
      pred <- predict(model, X_test, probability = (length(unique(y_train)) == 2))
      prob <- NULL
      if (length(unique(y_train)) == 2) {
        prob_attr <- attr(pred, "probabilities")
        prob <- prob_attr[, levels(y_train)[2]]
      }
      eval_model(as.factor(pred), prob, "SVM", y_test, fold)
    }, error = function(e) {
      message("SVM Error: ", e$message)
    })
    
    # -------------------------------
    # XGBoost
    # -------------------------------
    # 
    # tryCatch({
    #   X_combined <- rbind(X_train, X_test)
    #   X_mat <- model.matrix(~ . - 1, data = X_combined)
    #   X_train_xgb <- X_mat[1:nrow(X_train), ]
    #   X_test_xgb <- X_mat[(nrow(X_train)+1):nrow(X_mat), ]
    #   y_train_xgb <- as.numeric(y_train) - 1
    #   y_test_xgb  <- as.numeric(y_test) - 1
    #   num_classes <- length(levels(y_train))
    #   obj <- if (num_classes > 2) "multi:softmax" else "binary:logistic"
    # 
    #   params <- list(objective = obj, num_class = if (num_classes > 2) num_classes else NULL)
    #   model <- xgboost(data = X_train_xgb, label = y_train_xgb, nrounds = 50, params = params, verbose = 0)
    #   pred_prob <- predict(model, X_test_xgb)
    #   pred <- if (num_classes > 2) levels(y_train)[pred_prob + 1] else ifelse(pred_prob > 0.5, levels(y_train)[2], levels(y_train)[1])
    #   prob <- if (num_classes == 2) pred_prob else NULL
    #   eval_model(as.factor(pred), prob, "XGBoost", y_test, group_label)
    # }, error = function(e) {
    #   message("XGBoost Error: ", e$message)
    # })
    
    #-----------------------------------
    # Average performance across k-folds
    #-----------------------------------
    
    for (model_name in names(model_metrics)) {
      df <- bind_rows(model_metrics[[model_name]])
      avg <- colMeans(df, na.rm = TRUE)
      results_list[[length(results_list) + 1]] <<- data.frame(
        Group = group_label,
        Model = model_name,
        Accuracy = round(avg["Accuracy"], 3),
        Precision = round(avg["Precision"], 3),
        Recall = round(avg["Recall"], 3),
        AUC = round(avg["AUC"], 3)
      )
    }
  }
  
  original_levels <- sort(unique(as.numeric(as.character(data[[target_col]]))))
  n_levels <- length(original_levels)
  
  #-------------------------------------
  # 7-classes
  #-------------------------------------
  
  cat("\n====== 7-class Model ======\n")
  run_models(data, target_col, "7-class")
  
  #-------------------------------------
  # 3-classes
  #-------------------------------------
  
  cat("\n====== 3-class Groupings ======\n")
  for (i in 1:(n_levels - 2)) {
    for (j in (i + 1):(n_levels - 1)) {
      g1 <- original_levels[1:i]
      g2 <- original_levels[(i + 1):j]
      g3 <- original_levels[(j + 1):n_levels]
      if (length(g1) > 0 && length(g2) > 0 && length(g3) > 0) {
        group_label <- sprintf("3-class (%s)/(%s)/(%s)",
                               paste(g1, collapse = ","), paste(g2, collapse = ","), paste(g3, collapse = ","))
        grouped_data <- data %>% select(-all_of(target_col))
        grouped_data[[paste0(target_col, "_3")]] <- as.factor(
          ifelse(as.numeric(as.character(data[[target_col]])) %in% g1, "Low",
                 ifelse(as.numeric(as.character(data[[target_col]])) %in% g2, "Medium", "High"))
        )
        run_models(grouped_data, paste0(target_col, "_3"), group_label)
      }
    }
  }
  
  #-------------------------------------
  # 2-classes
  #-------------------------------------
  
  cat("\n====== All 2-class Groupings ======\n")
  for (i in 1:(n_levels - 1)) {
    g1 <- original_levels[1:i]
    g2 <- original_levels[(i + 1):n_levels]
    if (length(g1) > 0 && length(g2) > 0) {
      group_label <- sprintf("2-class (%s) vs (%s)", paste(g1, collapse = ","), paste(g2, collapse = ","))
      grouped_data <- data %>% select(-all_of(target_col))
      grouped_data[[paste0(target_col, "_2")]] <- as.factor(
        ifelse(as.numeric(as.character(data[[target_col]])) %in% g1, "Low", "High")
      )
      run_models(grouped_data, paste0(target_col, "_2"), group_label)
    }
  }
  
  #-------------------------------------
  # Save results
  #-------------------------------------
  
  results_df <- bind_rows(results_list)
  write.csv(results_df, 
            file = file.path("results", paste0("model_performance_results_", target_col, ".csv")), 
            row.names = FALSE)
  message(paste0("✅ All results saved to 'model_performance_results_", target_col, ".csv'"))
  
}

#--------------------------------------------#
#------ performance function ends here -----#
#--------------------------------------------#



#--------------------------------------------------------------------------------
# Now get the results for all work load variables
#--------------------------------------------------------------------------------

final_data_for_analysis <- read_excel("data/final_data_for_analysis.xlsx")

#---------------------------------------------------------------------------------
# Data Pre-processing
#---------------------------------------------------------------------------------

# clean column names
final_data_for_analysis <- janitor::clean_names(final_data_for_analysis)

# Round wl_total to nearest integer
final_data_for_analysis$wl_total <- round(final_data_for_analysis$wl_total)
final_data_for_analysis$wl_total <- pmin(pmax(final_data_for_analysis$wl_total, 1), 7)
final_data_for_analysis$wl_total <- as.factor(final_data_for_analysis$wl_total)


# List of workload target variables
target_vars <- c(
  "wl_physical", "wl_mental", "wl_temporal",
  "wl_performance", "wl_effort", "wl_frustration", "wl_total"
)


# Common feature columns
feature_cols <- c(
  "los" = "length_of_patient_stay_hrs",
  "role", "days_patient_seen", "rvu_total", "patient_contact_time", 
  "complexity_type", "complexity", "chart_review_activities", 
  "current_cls", "max_cls", 
  "CCIS" = "charlson_comorbidity_index_score_column", 
  "ACH_score" = "ach_eligibility_v3_score", 
  "workload_acuity_score"
)

# Run for each target variable
for (var in target_vars) {
  cat("\n\n===========================\n")
  cat("Running analysis for:", var, "\n")
  cat("===========================\n")
  
  # Dynamically select columns including the current target
  data_for_model <- final_data_for_analysis %>% 
    select(all_of(c(var, feature_cols)))
  
  # Drop NAs
  data <- data_for_model %>% drop_na()
  
  # Convert target to factor
  data[[var]] <- as.factor(data[[var]])
  
  # Run analysis (uses the var name as output identifier)
  run_wl_prediction_analysis(data, var)
}

