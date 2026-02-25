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
library(binom)
library(kknn)
library(pROC)
library(readxl)
library(janitor)

#--------------------------------------------------------------------------------
# function to calculate Accuracy, precision, recall, AUC with 95% CI
#--------------------------------------------------------------------------------
run_wl_prediction_analysis <- function(data, target_col) {
  
  results_list <- list()
  k_folds <- 10
  
  run_models <- function(data, y_colname, group_label) {
    
    set.seed(555)
    data[[y_colname]] <- as.factor(data[[y_colname]])
    data <- data[!is.na(data[[y_colname]]), , drop = FALSE]
    folds <- createFolds(data[[y_colname]], k = k_folds, list = TRUE, returnTrain = FALSE)
    
    model_metrics <- list()
    all_preds <- list()
    
    for (fold in 1:k_folds) {
      test_idx <- folds[[fold]]
      train <- data[-test_idx, ] %>% droplevels()
      test  <- data[test_idx, ] %>% droplevels()
      
      X_train <- train %>% select(-all_of(y_colname))
      X_test  <- test %>% select(-all_of(y_colname))
      
      for (col in names(X_train)) {
        if (is.factor(X_train[[col]])) {
          X_train[[col]] <- factor(X_train[[col]])
          X_test[[col]] <- factor(X_test[[col]], levels = levels(X_train[[col]]))
        }
      }
      
      y_train <- factor(train[[y_colname]])
      y_test  <- factor(test[[y_colname]], levels = levels(y_train))
      
      eval_model <- function(pred, prob = NULL, method = "", y_test, fold_index) {
        # Ensure predicted factor matches true factor levels
        y_pred <- factor(pred, levels = levels(y_test))
        
        cm <- confusionMatrix(y_pred, y_test)
        acc <- cm$overall["Accuracy"]
        
        if (is.matrix(cm$byClass)) {
          prec <- mean(cm$byClass[, "Precision"], na.rm = TRUE)
          rec  <- mean(cm$byClass[, "Recall"], na.rm = TRUE)
        } else {
          prec <- cm$byClass["Precision"]
          rec  <- cm$byClass["Recall"]
        }
        
        # Compute TP, TN, FP, FN
        if (length(levels(y_test)) == 2) {
          # Binary classification
          TP <- cm$table[2, 2]
          TN <- cm$table[1, 1]
          FP <- cm$table[1, 2]
          FN <- cm$table[2, 1]
        } else {
          # Multiclass: One-vs-rest for each class
          TP <- FP <- FN <- TN <- numeric(length(levels(y_test)))
          names(TP) <- names(FP) <- names(FN) <- names(TN) <- levels(y_test)
          
          for (cls in levels(y_test)) {
            y_true_bin <- ifelse(y_test == cls, 1, 0)
            y_pred_bin <- ifelse(y_pred == cls, 1, 0)
            
            TP[cls] <- sum(y_true_bin == 1 & y_pred_bin == 1)
            TN[cls] <- sum(y_true_bin == 0 & y_pred_bin == 0)
            FP[cls] <- sum(y_true_bin == 0 & y_pred_bin == 1)
            FN[cls] <- sum(y_true_bin == 1 & y_pred_bin == 0)
          }
          
          # Optionally, take mean across classes for overall multiclass metric
          TP <- mean(TP)
          TN <- mean(TN)
          FP <- mean(FP)
          FN <- mean(FN)
        }
        
        auc <- NA
        if (!is.null(prob)) {
          if (length(levels(y_test)) == 2) {
            roc_obj <- tryCatch({
              roc(y_test, prob, levels = levels(y_test))
            }, error = function(e) NULL)
            if (!is.null(roc_obj)) auc <- as.numeric(auc(roc_obj))
          } else {
            if (is.data.frame(prob) || is.matrix(prob)) {
              y_bin <- model.matrix(~ y_test - 1)
              aucs <- c()
              for (k in 1:ncol(y_bin)) {
                roc_obj <- tryCatch({
                  roc(y_bin[, k], prob[, k], quiet = TRUE)
                }, error = function(e) NULL)
                if (!is.null(roc_obj)) aucs <- c(aucs, as.numeric(auc(roc_obj)))
              }
              if (length(aucs) > 0) auc <- mean(aucs, na.rm = TRUE)
            }
          }
        }
        
        model_metrics[[method]][[fold_index]] <<- data.frame(
          Accuracy = acc,
          Precision = prec,
          Recall = rec,
          AUC = auc,
          TP = TP,
          TN = TN,
          FP = FP,
          FN = FN
        )
        
        all_preds[[method]][[fold_index]] <<- list(
          y_true = y_test,
          y_pred = y_pred,
          y_prob = prob
        )
      }
      
      # #-------------------------------
      # # Logistic / Multinomial
      # #-------------------------------
      # 
      # 
      # tryCatch({
      #   if (length(unique(y_train)) > 2) {
      #     model <- multinom(as.formula(paste(y_colname, "~ .")), data = train, trace = FALSE)
      #     prob <- predict(model, newdata = X_test, type = "probs")
      #     if (is.vector(prob)) {
      #       prob <- cbind(1 - prob, prob)
      #       colnames(prob) <- levels(y_train)
      #     }
      #     for (cls in levels(y_train)) {
      #       if (!cls %in% colnames(prob)) {
      #         prob <- cbind(prob, setNames(rep(0, nrow(prob)), cls))
      #       }
      #     }
      #     prob <- prob[, levels(y_train), drop = FALSE]
      #     pred <- predict(model, newdata = X_test)
      #     eval_model(pred, prob, "Multinomial Logistic", y_test, fold)
      #   } else {
      #     model <- glm(y_train ~ ., data = cbind(X_train, y_train), family = "binomial")
      #     prob <- predict(model, newdata = X_test, type = "response")
      #     pred <- ifelse(prob > 0.5, levels(y_train)[2], levels(y_train)[1])
      #     eval_model(pred, prob, "Logistic Regression", y_test, fold)
      #   }
      # }, error = function(e) {
      #   message("Logistic Error: ", e$message)
      # })
      # 
      # #-------------------------------
      # # Random Forest
      # #-------------------------------
      # 
      # tryCatch({
      #   model <- randomForest(as.formula(paste(y_colname, "~ .")), data = train)
      #   pred <- predict(model, newdata = X_test)
      # 
      #   prob <- predict(model, newdata = X_test, type = "prob")
      #   if (length(levels(y_train)) == 2) {
      #     prob <- prob[, 2]  # binary: probability for positive class
      #   }
      # 
      #   eval_model(as.factor(pred), prob, "Random Forest", y_test, fold)
      # }, error = function(e) {
      #   message("Random Forest Error: ", e$message)
      # })
      # 
      # #-------------------------------
      # # KNN
      # #-------------------------------
      # 
      # tryCatch({
      #   # 1. Convert all predictors to numeric via model.matrix
      #   X_train_num <- model.matrix(~ . - 1, data = X_train)
      #   X_test_num  <- model.matrix(~ . - 1, data = X_test)
      # 
      #   # 2. Scale numeric predictors
      #   X_combined <- rbind(X_train_num, X_test_num)
      #   X_scaled <- scale(X_combined)
      #   train_scaled <- X_scaled[1:nrow(X_train_num), , drop = FALSE]
      #   test_scaled  <- X_scaled[(nrow(X_train_num)+1):nrow(X_scaled), , drop = FALSE]
      # 
      #   # 3. Prepare data frames with target
      #   train_df <- data.frame(train_scaled, y_train = y_train)
      #   test_df  <- data.frame(test_scaled)
      # 
      #   # 4. Choose k dynamically to avoid small-class issues
      #   k_val <- min(5, min(table(y_train)))
      # 
      #   # 5. Fit KNN
      #   knn_fit <- kknn(
      #     formula = y_train ~ .,
      #     train = train_df,
      #     test  = test_df,
      #     k = k_val,
      #     distance = 2,
      #     kernel = "rectangular"
      #   )
      # 
      #   # 6. Predicted class labels
      #   pred <- fitted(knn_fit)
      # 
      #   # 7. Extract probabilities
      #   if (length(levels(y_train)) == 2) {
      #     # For binary: probability of positive class
      #     pred_prob <- knn_fit$prob
      #     if (is.matrix(pred_prob)) {
      #       pred_prob <- pred_prob[, 2]
      #     }
      #   } else {
      #     # Multiclass: probability matrix
      #     pred_prob <- knn_fit$prob
      #   }
      # 
      #   # 8. Evaluate
      #   eval_model(as.factor(pred), pred_prob, "KNN", y_test, fold)
      # 
      # }, error = function(e) {
      #   message("KNN Error: ", e$message)
      # })
      # 
      # tryCatch({
      #   # Combine train and test for scaling
      #   X_mat <- model.matrix(~ . - 1, data = rbind(X_train, X_test))
      #   X_scaled <- scale(X_mat)
      #   train_scaled <- X_scaled[1:nrow(X_train), ]
      #   test_scaled  <- X_scaled[(nrow(X_train) + 1):nrow(X_scaled), ]
      # 
      #   # Prepare data frames with target variable
      #   train_df <- data.frame(X_train, y_train = y_train)
      #   test_df  <- data.frame(X_test)
      # 
      #   # Fit KNN using kknn (no 'prob' argument)
      #   knn_fit <- kknn(
      #     formula = as.formula("y_train ~ ."),
      #     train = train_df,
      #     test  = test_df,
      #     k = 5,
      #     distance = 2,         # optional: Euclidean distance
      #     kernel = "rectangular" # default uniform weights
      #   )
      # 
      #   pred <- fitted(knn_fit)        # predicted class labels
      #   pred_prob <- knn_fit$prob      # probability matrix for all classes
      # 
      #   eval_model(as.factor(pred), pred_prob, "KNN", y_test, fold)
      # 
      # }, error = function(e) {
      #   message("KNN Error: ", e$message)
      # })
      # 
      # #-------------------------------
      # # SVM
      # #-------------------------------
      # 
      # tryCatch({
      #   # Always enable probability = TRUE
      #   model <- svm(as.formula(paste(y_colname, "~ .")), data = train, probability = TRUE)
      #   
      #   pred <- predict(model, X_test, probability = TRUE)
      #   
      #   # Extract probabilities for all classes
      #   prob_attr <- attr(pred, "probabilities")
      #   prob <- NULL
      #   if (!is.null(prob_attr)) {
      #     if (length(levels(y_train)) == 2) {
      #       # Binary: positive class
      #       prob <- prob_attr[, levels(y_train)[2]]
      #       # Ensure probability has variation
      #       if (length(unique(prob)) == 1) prob <- NULL
      #     } else {
      #       # Multiclass: keep full probability matrix
      #       prob <- prob_attr[, levels(y_train), drop = FALSE]
      #     }
      #   }
      #   
      #   # Evaluate
      #   eval_model(as.factor(pred), prob, "SVM", y_test, fold)
      #   
      # }, error = function(e) {
      #   message("SVM Error: ", e$message)
      # })
      
      #-------------------------------
      # XGBoost
      #-------------------------------
      tryCatch({
        # Combine train and test for creating model matrix
        X_combined <- rbind(X_train, X_test)
        X_mat <- model.matrix(~ . - 1, data = X_combined)
        
        X_train_xgb <- X_mat[1:nrow(X_train), ]
        X_test_xgb  <- X_mat[(nrow(X_train) + 1):nrow(X_mat), ]
        
        # Ensure binary classes are 0 and 1 in correct order
        if (length(levels(y_train)) == 2) {
          y_train_xgb <- ifelse(y_train == levels(y_train)[2], 1, 0)
          y_test_xgb  <- ifelse(y_test == levels(y_train)[2], 1, 0)
        } else {
          y_train_xgb <- as.numeric(y_train) - 1
          y_test_xgb  <- as.numeric(y_test) - 1
        }
        
        num_classes <- length(levels(y_train))
        if (num_classes > 2) {
          obj <- "multi:softprob"
          params <- list(objective = obj, num_class = num_classes, eval_metric = "mlogloss")
        } else {
          obj <- "binary:logistic"
          params <- list(objective = obj, eval_metric = "logloss")
        }
        
        model <- xgboost(
          data = X_train_xgb,
          label = y_train_xgb,
          nrounds = 50,
          params = params,
          verbose = 0,
          nthread = 4  # <--- limit CPU usage
        )
        
        # Predictions
        pred_prob <- predict(model, X_test_xgb)
        
        if (num_classes == 2) {
          pred <- ifelse(pred_prob > 0.5, levels(y_train)[2], levels(y_train)[1])
          prob <- pred_prob
        } else {
          # multiclass: reshape vector to matrix of nrow(test) x num_classes
          pred_prob_mat <- matrix(pred_prob, ncol = num_classes, byrow = TRUE)
          colnames(pred_prob_mat) <- levels(y_train)
          pred <- levels(y_train)[max.col(pred_prob_mat)]  # class with highest probability
          prob <- pred_prob_mat
        }
        
        # Evaluate
        eval_model(as.factor(pred), prob, "XGBoost", y_test, fold)
        
      }, error = function(e) {
        message("XGBoost Error: ", e$message)
      })
      
      
    }
    
    # Compute means + 95% CIs
    for (model_name in names(model_metrics)) {
      df <- bind_rows(model_metrics[[model_name]])
      
      ci_acc <- binom.confint(
        sum(unlist(lapply(all_preds[[model_name]], function(x) sum(x$y_true == x$y_pred)))),
        length(unlist(lapply(all_preds[[model_name]], function(x) x$y_true))),
        conf.level = 0.95, methods = "wilson"
      )
      
      # Precision and Recall CI using quantiles of fold-level values
      prec_vals <- df$Precision[!is.na(df$Precision)]
      prec_ci <- quantile(prec_vals, probs = c(0.025, 0.975), na.rm = TRUE)
      
      rec_vals <- df$Recall[!is.na(df$Recall)]
      rec_ci <- quantile(rec_vals, probs = c(0.025, 0.975), na.rm = TRUE)     
      
      auc_vals <- df$AUC[!is.na(df$AUC)]
      if (length(auc_vals) >= 1) {  # at least one valid AUC
        auc_ci <- if (length(auc_vals) > 1) {
          quantile(auc_vals, probs = c(0.025, 0.975), na.rm = TRUE)
        } else {
          c(auc_vals, auc_vals)  # use single value for CI
        }
      } else {
        auc_ci <- c(NA, NA)
      }
      # auc_ci <- c(NA, NA)
      # if (length(auc_vals) > 1) {
      #   auc_ci <- quantile(auc_vals, probs = c(0.025, 0.975), na.rm = TRUE)
      # }
      
      results_list[[length(results_list) + 1]] <<- data.frame(
        Group = group_label,
        Model = model_name,
        Accuracy = round(mean(df$Accuracy, na.rm = TRUE), 3),
        Accuracy_CI = sprintf("[%.3f, %.3f]", ci_acc$lower, ci_acc$upper),
        Precision = round(mean(df$Precision, na.rm = TRUE), 3),
        Precision_CI = sprintf("[%.3f, %.3f]", prec_ci[1], prec_ci[2]),
        Recall = round(mean(df$Recall, na.rm = TRUE), 3),
        Recall_CI = sprintf("[%.3f, %.3f]", rec_ci[1], rec_ci[2]),
        AUC = round(mean(df$AUC, na.rm = TRUE), 3),
        AUC_CI = ifelse(!all(is.na(auc_ci)),
                        sprintf("[%.3f, %.3f]", auc_ci[1], auc_ci[2]), NA),
        TP = as.integer(round(mean(df$TP, na.rm = TRUE), 1)),
        TN = as.integer(round(mean(df$TN, na.rm = TRUE), 1)),
        FP = as.integer(round(mean(df$FP, na.rm = TRUE), 1)),
        FN = as.integer(round(mean(df$FN, na.rm = TRUE), 1))
        
      )
    }
  }
  
  original_levels <- sort(unique(as.numeric(as.character(data[[target_col]]))))
  n_levels <- length(original_levels)
  
  cat("\n====== 7-class Model ======\n")
  run_models(data, target_col, "7-class")
  
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
  
  results_df <- bind_rows(results_list)
  write.csv(results_df, 
            file = file.path("results", paste0("model_performance_results_", target_col, ".csv")), 
            row.names = FALSE)
  message(paste0("✅ All results saved to 'model_performance_results_", target_col, ".csv'"))
}

#--------------------------------------------------------------------------------
# Run analysis for workload variables
#--------------------------------------------------------------------------------
final_data_for_analysis <- read_excel("data/final_data_for_analysis.xlsx")
final_data_for_analysis <- janitor::clean_names(final_data_for_analysis)
final_data_for_analysis$wl_total <- round(final_data_for_analysis$wl_total)
final_data_for_analysis$wl_total <- pmin(pmax(final_data_for_analysis$wl_total, 1), 7)
final_data_for_analysis$wl_total <- as.factor(final_data_for_analysis$wl_total)

target_vars <- "wl_physical"

feature_cols <- c(
  "los" = "length_of_patient_stay_hrs",
  "role", "days_patient_seen", "rvu_total", "patient_contact_time", 
  "complexity_type", "complexity", "chart_review_activities", 
  "current_cls", "max_cls", 
  "CCIS" = "charlson_comorbidity_index_score_column", 
  "ACH_score" = "ach_eligibility_v3_score", 
  "workload_acuity_score"
)

for (var in target_vars) {
  cat("\n\n===========================\n")
  cat("Running analysis for:", var, "\n")
  cat("===========================\n")
  
  data_for_model <- final_data_for_analysis %>% 
    select(all_of(c(var, feature_cols)))
  
  data <- data_for_model %>% drop_na()
  data[[var]] <- as.factor(data[[var]])
  
  run_wl_prediction_analysis(data, var)
}
