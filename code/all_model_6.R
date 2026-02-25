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
library(kknn)
library(binom)
library(pROC)
library(readxl)
library(janitor)

#--------------------------------------------------------------------------------
# Function to calculate Accuracy, Precision, Recall, AUC with 95% CI
#--------------------------------------------------------------------------------
run_wl_prediction_analysis <- function(data, target_col) {
  
  results_list <- list()
  k_folds <- 5  # Number of cross-validation folds
  
  run_models <- function(data, y_colname, group_label) {
    
    set.seed(555)
    data[[y_colname]] <- as.factor(data[[y_colname]])
    data <- data[!is.na(data[[y_colname]]), , drop = FALSE]
    folds <- createFolds(data[[y_colname]], k = k_folds, list = TRUE, returnTrain = FALSE)
    
    model_metrics <- list()
    all_preds <- list()
    
    # Evaluate predictions function
    eval_model <- function(pred, prob = NULL, method = "", y_test, fold_index) {
      y_pred <- factor(pred, levels = levels(y_test))
      cm <- confusionMatrix(y_pred, y_test)
      acc <- cm$overall["Accuracy"]
      
      # Handle multiclass precision and recall
      if (is.matrix(cm$byClass)) {
        prec <- if (all(is.na(cm$byClass[, "Precision"]))) 0 else mean(cm$byClass[, "Precision"], na.rm = TRUE)
        rec  <- if (all(is.na(cm$byClass[, "Recall"])))    0 else mean(cm$byClass[, "Recall"], na.rm = TRUE)
        
      } else {
        prec <- cm$byClass["Precision"]
        rec  <- cm$byClass["Recall"]
      }
      
      # Compute TP, TN, FP, FN
      if (length(levels(y_test)) == 2) {
        TP <- cm$table[2,2]; TN <- cm$table[1,1]
        FP <- cm$table[1,2]; FN <- cm$table[2,1]
      } else {
        TP <- TN <- FP <- FN <- numeric(length(levels(y_test)))
        names(TP) <- levels(y_test)
        for (cls in levels(y_test)) {
          y_true_bin <- ifelse(y_test == cls, 1, 0)
          y_pred_bin <- ifelse(y_pred == cls, 1, 0)
          TP[cls] <- sum(y_true_bin & y_pred_bin)
          TN[cls] <- sum(!y_true_bin & !y_pred_bin)
          FP[cls] <- sum(!y_true_bin & y_pred_bin)
          FN[cls] <- sum(y_true_bin & !y_pred_bin)
        }
        TP <- round(mean(TP)); TN <- round(mean(TN)); FP <- round(mean(FP)); FN <- round(mean(FN))
      }
      
      # Compute AUC if probabilities are available
      auc <- NA
      if (!is.null(prob)) {
        if (length(levels(y_test)) == 2) {
          roc_obj <- tryCatch(roc(y_test, prob, levels = levels(y_test), quiet = TRUE), error = function(e) NULL)
          if (!is.null(roc_obj)) auc <- as.numeric(auc(roc_obj))
        } else if (is.matrix(prob) || is.data.frame(prob)) {
          y_bin <- model.matrix(~ y_test - 1)
          aucs <- sapply(1:ncol(y_bin), function(k) {
            tryCatch({
              roc_obj <- roc(y_bin[,k], prob[,k], quiet = TRUE)
              as.numeric(auc(roc_obj))
            }, error = function(e) NA)
          })
          auc <- if (all(is.na(aucs))) NA else mean(aucs, na.rm = TRUE)
        }
      }
      
      
      # Store metrics
      model_metrics[[method]][[fold_index]] <<- data.frame(
        Accuracy = acc, Precision = prec, Recall = rec, AUC = auc,
        TP = TP, TN = TN, FP = FP, FN = FN
      )
      all_preds[[method]][[fold_index]] <<- list(y_true = y_test, y_pred = y_pred, y_prob = prob)
    }
    
    # Loop through folds
    for (fold in 1:k_folds) {
      test_idx <- folds[[fold]]
      train <- data[-test_idx, ] %>% droplevels()
      test  <- data[test_idx, ] %>% droplevels()
      
      X_train <- train %>% select(-all_of(y_colname))
      X_test  <- test %>% select(-all_of(y_colname))
      
      # Ensure factor levels are consistent
      for (col in names(X_train)) {
        if (is.factor(X_train[[col]])) {
          X_test[[col]] <- factor(X_test[[col]], levels = levels(X_train[[col]]))
        }
      }
      
      full_levels <- levels(data[[y_colname]])
      y_train <- factor(train[[y_colname]], levels = full_levels)
      y_test  <- factor(test[[y_colname]],  levels = full_levels)
      
      #-------------------------------
      # Logistic / Multinomial
      #-------------------------------
      tryCatch({
        if (length(unique(y_train)) > 2) {
          model <- multinom(as.formula(paste(y_colname, "~ .")), data = train, trace = FALSE)
          prob <- predict(model, newdata = X_test, type = "probs")
          pred <- predict(model, newdata = X_test)
          pred <- factor(pred, levels = full_levels)
          eval_model(pred, prob, "Multinomial Logistic", y_test, fold)
        } else {
          model <- glm(y_train ~ ., data = cbind(X_train, y_train), family = "binomial")
          prob <- predict(model, newdata = X_test, type = "response")
          pred <- ifelse(prob > 0.5, levels(y_train)[2], levels(y_train)[1])
          eval_model(pred, prob, "Logistic Regression", y_test, fold)
        }
      }, error = function(e) message("Logistic Error: ", e$message))
      
      # #-------------------------------
      # # KNN
      # #-------------------------------
      # tryCatch({
      #   X_train_num <- model.matrix(~ . -1, data = X_train)
      #   X_test_num  <- model.matrix(~ . -1, data = X_test)
      #   
      #   # Fix column mismatch
      #   missing_cols <- setdiff(colnames(X_train_num), colnames(X_test_num))
      #   if(length(missing_cols) > 0) {
      #     X_test_num <- cbind(
      #       X_test_num, 
      #       matrix(0, nrow=nrow(X_test_num), ncol=length(missing_cols),
      #              dimnames = list(NULL, missing_cols))
      #     )
      #   }
      #   X_test_num <- X_test_num[, colnames(X_train_num)]
      #   
      #   X_scaled <- scale(rbind(X_train_num, X_test_num))
      #   train_scaled <- X_scaled[1:nrow(X_train_num), ]
      #   test_scaled  <- X_scaled[(nrow(X_train_num)+1):nrow(X_scaled), ]
      #   
      #   knn_fit <- kknn(
      #     y_train ~ ., 
      #     train = data.frame(train_scaled, y_train), 
      #     test  = data.frame(test_scaled), 
      #     k = min(5, min(table(y_train))), 
      #     distance = 2, 
      #     kernel = "rectangular"
      #   )
      #   
      #   pred <- fitted(knn_fit)
      #   pred <- factor(pred, levels = full_levels)
      #   
      #   # Extract probabilities
      #   prob <- knn_fit$prob
      #   if (is.matrix(prob) && ncol(prob) > 1) {
      #     # For binary classification, pick positive class (second level)
      #     if (length(full_levels) == 2) {
      #       prob <- prob[, full_levels[2]]
      #     } else {
      #       # For multi-class, keep full probability matrix
      #       colnames(prob) <- full_levels
      #     }
      #   }
      #   
      #   eval_model(pred, prob, "KNN", y_test, fold)
      # }, error = function(e) message("KNN Error: ", e$message))
      # #-------------------------------
      # # Random Forest
      # #-------------------------------
      # tryCatch({
      #   model <- randomForest(as.formula(paste(y_colname, "~ .")), data = train)
      #   pred <- predict(model, X_test)
      #   prob <- if (length(levels(y_train)) == 2) predict(model, X_test, type = "prob")[,2] else predict(model, X_test, type = "prob")
      #   eval_model(pred, prob, "Random Forest", y_test, fold)
      # }, error = function(e) message("Random Forest Error: ", e$message))
      # #-------------------------------
      # # SVM
      # #-------------------------------
      # tryCatch({
      #   model <- svm(as.formula(paste(y_colname, "~ .")), data = train, probability = TRUE)
      #   pred <- predict(model, X_test, probability = TRUE)
      #   prob_attr <- attr(pred, "probabilities")
      #   prob <- if (!is.null(prob_attr)) {
      #     if (length(levels(y_train)) == 2) prob_attr[, levels(y_train)[2]] else prob_attr[, levels(y_train), drop = FALSE]
      #   }
      #   eval_model(pred, prob, "SVM", y_test, fold)
      # }, error = function(e) message("SVM Error: ", e$message))
      # 
      # #-------------------------------
      # # XGBoost
      # #-------------------------------
      # tryCatch({
      #   X_mat <- model.matrix(~ . -1, data = rbind(X_train, X_test))
      #   X_train_xgb <- X_mat[1:nrow(X_train), ]
      #   X_test_xgb  <- X_mat[(nrow(X_train)+1):nrow(X_mat), ]
      #   if (length(levels(y_train)) == 2) {
      #     y_train_xgb <- ifelse(y_train == levels(y_train)[2], 1, 0)
      #     y_test_xgb  <- ifelse(y_test == levels(y_train)[2], 1, 0)
      #   } else {
      #     y_train_xgb <- as.numeric(y_train)-1
      #     y_test_xgb  <- as.numeric(y_test)-1
      #   }
      #   num_classes <- length(levels(y_train))
      #   params <- if (num_classes > 2) list(objective = "multi:softprob", num_class = num_classes, eval_metric = "mlogloss") else list(objective = "binary:logistic", eval_metric = "logloss")
      #   model <- xgboost(data = X_train_xgb, label = y_train_xgb, nrounds = 50, params = params, verbose = 0, nthread = 4)
      #   pred_prob <- predict(model, X_test_xgb)
      #   if (num_classes == 2) {
      #     pred <- ifelse(pred_prob > 0.5, levels(y_train)[2], levels(y_train)[1])
      #     prob <- pred_prob
      #   } else {
      #     pred_prob_mat <- matrix(pred_prob, ncol = num_classes, byrow = TRUE)
      #     colnames(pred_prob_mat) <- levels(y_train)
      #     pred <- levels(y_train)[max.col(pred_prob_mat)]
      #     prob <- pred_prob_mat
      #   }
      #   eval_model(pred, prob, "XGBoost", y_test, fold)
      # }, error = function(e) message("XGBoost Error: ", e$message))
      
    }
    
    # Compute metrics with 95% CI across folds but keep TP/TN/FP/FN from last fold
    for (model_name in names(model_metrics)) {
      df <- bind_rows(model_metrics[[model_name]])
      
      # Accuracy CI
      ci_acc <- binom.confint(
        sum(unlist(lapply(all_preds[[model_name]], function(x) sum(x$y_true == x$y_pred)))),
        length(unlist(lapply(all_preds[[model_name]], function(x) x$y_true))),
        conf.level = 0.95, methods = "wilson"
      )
      
      # Precision & Recall CI
      prec_ci <- quantile(df$Precision, probs = c(0.025, 0.975), na.rm = TRUE)
      rec_ci  <- quantile(df$Recall, probs = c(0.025, 0.975), na.rm = TRUE)
      
      # AUC CI
      auc_vals <- df$AUC[!is.na(df$AUC)]
      auc_ci <- if (length(auc_vals) > 1) quantile(auc_vals, probs = c(0.025, 0.975)) else c(auc_vals, auc_vals)
      
      # TP/TN/FP/FN from last successful fold
      last_fold_df <- tail(model_metrics[[model_name]], 1)[[1]]
      
      results_list[[length(results_list)+1]] <<- data.frame(
        Group = group_label,
        Model = model_name,
        Accuracy = round(mean(df$Accuracy, na.rm = TRUE), 3),
        Accuracy_CI = sprintf("[%.3f, %.3f]", ci_acc$lower, ci_acc$upper),
        Precision = round(mean(df$Precision, na.rm = TRUE), 3),
        Precision_CI = sprintf("[%.3f, %.3f]", prec_ci[1], prec_ci[2]),
        Recall = round(mean(df$Recall, na.rm = TRUE), 3),
        Recall_CI = sprintf("[%.3f, %.3f]", rec_ci[1], rec_ci[2]),
        AUC = round(mean(df$AUC, na.rm = TRUE), 3),
        AUC_CI = ifelse(!all(is.na(auc_ci)), sprintf("[%.3f, %.3f]", auc_ci[1], auc_ci[2]), NA),
        TP = last_fold_df$TP,
        TN = last_fold_df$TN,
        FP = last_fold_df$FP,
        FN = last_fold_df$FN
      )
    }
  }
  
  
  
  #-------------------------------
  # Run analysis for 7-class, 3-class, 2-class groupings
  #-------------------------------
  original_levels <- sort(unique(as.numeric(as.character(data[[target_col]]))))
  n_levels <- length(original_levels)
  
  cat("\n====== 7-class Model ======\n")
  run_models(data, target_col, "7-class")
  
  cat("\n====== 3-class Groupings ======\n")
  for (i in 1:(n_levels - 2)) {
    for (j in (i + 1):(n_levels - 1)) {
      g1 <- original_levels[1:i]; g2 <- original_levels[(i+1):j]; g3 <- original_levels[(j+1):n_levels]
      grouped_data <- data %>% select(-all_of(target_col))
      grouped_data[[paste0(target_col, "_3")]] <- factor(
        ifelse(as.numeric(as.character(data[[target_col]])) %in% g1, "Low",
               ifelse(as.numeric(as.character(data[[target_col]])) %in% g2, "Medium", "High"))
      )
      run_models(grouped_data, paste0(target_col, "_3"),
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
    run_models(grouped_data, paste0(target_col, "_2"),
               sprintf("2-class (%s) vs (%s)", paste(g1, collapse=","), paste(g2, collapse=",")))
  }
  
  # Save results
  results_df <- bind_rows(results_list)
  write.csv(results_df,
            file = file.path("results", paste0("model_performance_results_", target_col, ".csv")),
            row.names = FALSE)
  message(paste0("✅ All results saved to 'model_performance_results_", target_col, ".csv'"))
}

#--------------------------------------------------------------------------------
# Run analysis for workload variables
#--------------------------------------------------------------------------------
final_data_for_analysis <- read_tsv("data/ModelingData.txt")
PCA_final_score_data <- read_csv("data/PCA_final_score_data.csv")
final_data_for_analysis <- cbind(final_data_for_analysis,PCA_final_score_data) %>% 
  janitor::clean_names() %>% 
  select("physical", "mental", "temporal",
    "performance", "effort", "frustration", "los","role", "dps", "rvu", "time",
    "complexity", "chart",
    "ccls", "mcls",
    "cci",
    "ach",
    "was", "workload_class"
  )

library(nnet)
library(caret)

final_data_for_analysis$workload_class <- factor(final_data_for_analysis$workload_class,
                                                 levels = c("Low", "Medium", "High"))
set.seed(123)
train_index <- createDataPartition(final_data_for_analysis$workload_class, p = 0.8, list = FALSE)
train_data <- final_data_for_analysis[train_index, ]
test_data  <- final_data_for_analysis[-train_index, ]

model <- multinom(workload_class ~ physical + mental + temporal + performance +
                    effort + frustration + los + role + dps + rvu + time +
                    complexity + chart + ccls + mcls + cci + ach + was,
                  data = train_data)


library(nnet)
library(caret)
library(binom)
library(dplyr)


set.seed(123)
folds <- createFolds(final_data_for_analysis$workload_class, k = 5, list = TRUE)

train_folds <- lapply(folds, function(idx) final_data_for_analysis[-idx, ])
test_folds  <- lapply(folds, function(idx) final_data_for_analysis[idx, ])

results_list <- list()

# Suppose you have cross-validation folds stored in a list
# train_folds[[i]] and test_folds[[i]]
for (fold in 1:length(train_folds)) {
  
  train <- train_folds[[fold]]
  test  <- test_folds[[fold]]
  
  # Fit multinomial logistic model
  model <- multinom(workload_class ~ ., data = train, trace = FALSE)
  
  # Predictions
  pred_class <- predict(model, newdata = test)
  pred_prob  <- predict(model, newdata = test, type = "prob")
  
  # Confusion matrix
  cm <- confusionMatrix(pred_class, test$workload_class)
  
  # Accuracy
  acc <- cm$overall["Accuracy"]
  
  # Precision / Recall / F1 (macro-averaged)
  prec <- mean(cm$byClass[,"Pos Pred Value"], na.rm = TRUE)
  rec  <- mean(cm$byClass[,"Sensitivity"], na.rm = TRUE)
  f1   <- 2 * (prec * rec) / (prec + rec)
  
  # AUC (macro-average)
  levels <- levels(test$workload_class)
  auc_vals <- sapply(levels, function(cl) {
    actual_binary <- ifelse(test$workload_class == cl, 1, 0)
    roc_obj <- pROC::roc(actual_binary, pred_prob[, cl])
    pROC::auc(roc_obj)
  })
  auc <- mean(auc_vals)
  
  # TP/TN/FP/FN per class (from last fold)
  last_cm <- cm$table
  TP <- diag(last_cm)
  FP <- colSums(last_cm) - TP
  FN <- rowSums(last_cm) - TP
  TN <- sum(last_cm) - TP - FP - FN
  
  results_list[[fold]] <- data.frame(
    Fold = fold,
    Accuracy = acc,
    Precision = prec,
    Recall = rec,
    F1 = f1,
    AUC = auc,
    TP = paste(TP, collapse=","),
    TN = paste(TN, collapse=","),
    FP = paste(FP, collapse=","),
    FN = paste(FN, collapse=",")
  )
}

# Combine all folds
df_metrics <- bind_rows(results_list)

# Compute mean ± 95% CI across folds
mean_ci <- df_metrics %>%
  summarise(
    Accuracy = mean(Accuracy, na.rm = TRUE),
    Accuracy_CI = paste0("[", round(quantile(Accuracy, 0.025),3), ", ", round(quantile(Accuracy, 0.975),3), "]"),
    Precision = mean(Precision, na.rm = TRUE),
    Precision_CI = paste0("[", round(quantile(Precision, 0.025),3), ", ", round(quantile(Precision, 0.975),3), "]"),
    Recall = mean(Recall, na.rm = TRUE),
    Recall_CI = paste0("[", round(quantile(Recall, 0.025),3), ", ", round(quantile(Recall, 0.975),3), "]"),
    F1 = mean(F1, na.rm = TRUE),
    F1_CI = paste0("[", round(quantile(F1, 0.025),3), ", ", round(quantile(F1, 0.975),3), "]"),
    AUC = mean(AUC, na.rm = TRUE),
    AUC_CI = paste0("[", round(quantile(AUC, 0.025),3), ", ", round(quantile(AUC, 0.975),3), "]"),
    TP = last(TP),
    TN = last(TN),
    FP = last(FP),
    FN = last(FN)
  )

mean_ci


# final_data_for_analysis <- read_excel("data/final_data_for_analysis.xlsx") %>%
#   janitor::clean_names() %>%
#   mutate(wl_total = pmin(pmax(round(wl_total), 1), 7) %>% as.factor())



# # List of workload target variables
# target_vars <- c("wl_physical", "wl_mental", "wl_temporal",
#                  "wl_performance", "wl_effort", "wl_frustration", "wl_total"
# )




for (var in target_vars) {
  cat("\n\n===========================\nRunning analysis for:", var, "\n===========================\n")
  data_for_model <- final_data_for_analysis %>% select(all_of(c(var, feature_cols))) %>% drop_na()
  data_for_model[[var]] <- as.factor(data_for_model[[var]])
  run_wl_prediction_analysis(data_for_model, var)
}
 