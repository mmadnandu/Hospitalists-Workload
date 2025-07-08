######################################################
#-----------------------------------------------------
#-----------------------------------------------------
# Predicting hospitalists' workloads
# MD  MOHIUDDIN ADNAN
#-----------------------------------------------------
#-----------------------------------------------------
######################################################

install.packages("xgboost")

library(usethis)
# git and github configuration
usethis::use_git()

#---------------------------------------------------------------------------------
# libraries
#---------------------------------------------------------------------------------

library(readxl)
library(janitor)
library(tidyverse)
library(caret)
library(nnet)
library(randomForest)
library(xgboost)
library(e1071)
library(class)
library(bit64)

#---------------------------------------------------------------------------------
# import data
#---------------------------------------------------------------------------------

final_data_for_analysis <- read_excel("/projects/cshcd/kern26/s312369.hosp.wl/adnan/Data/final_data_for_analysis.xlsx")
final_data_for_analysis <- janitor::clean_names(final_data_for_analysis)

data_for_model <- final_data_for_analysis %>% 
  select(wl_physical, los = length_of_patient_stay_hrs, role, days_patient_seen, rvu_total, patient_contact_time, complexity_type, 
         complexity, chart_review_activities, current_cls, max_cls, CCIS = charlson_comorbidity_index_score_column, ACH_score = ach_eligibility_v3_score, workload_acuity_score)


#---------------------------------------------------------------------------------
# Data Preprocessing
#---------------------------------------------------------------------------------

# Remove rows with missing values
data <- data_for_model %>% drop_na()

# Convert target to factor
data$wl_physical <- as.factor(data$wl_physical)

# Split into train/test
set.seed(123)
train_index <- createDataPartition(data$wl_physical, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data  <- data[-train_index, ]

# Prepare features and labels
X_train <- train_data %>% select(-wl_physical)
y_train <- train_data$wl_physical
X_test  <- test_data %>% select(-wl_physical)
y_test  <- test_data$wl_physical



##################################################################################
#---------------------------------------------------------------------------------
# 7 Classes
#---------------------------------------------------------------------------------
##################################################################################


#---------------------------------------------------------------------------------
# 1. Multinomial Logistic Regression (7 Classes)
#---------------------------------------------------------------------------------


mlr_model <- multinom(wl_physical ~ ., data = train_data)
mlr_pred <- predict(mlr_model, newdata = X_test)
confusionMatrix(mlr_pred, y_test)


#---------------------------------------------------------------------------------
# 2. Random Forest
#---------------------------------------------------------------------------------
rf_model <- randomForest(wl_physical ~ ., data = train_data, ntree = 100)
rf_pred <- predict(rf_model, newdata = X_test)
confusionMatrix(rf_pred, y_test)

#---------------------------------------------------------------------------------
# 3. XGBoost (requires numeric features)
#---------------------------------------------------------------------------------

# Convert to matrix
X_train_xgb <- model.matrix(~ . - 1, data = X_train)
X_test_xgb <- model.matrix(~ . - 1, data = X_test)
y_train_xgb <- as.numeric(as.character(y_train)) - 1  # 0-based
y_test_xgb <- as.numeric(as.character(y_test)) - 1

xgb_model <- xgboost(
  data = X_train_xgb,
  label = y_train_xgb,
  nrounds = 100,
  objective = "multi:softmax",
  num_class = length(unique(y_train))
)

xgb_pred <- predict(xgb_model, X_test_xgb)
confusionMatrix(as.factor(xgb_pred + 1), as.factor(y_test_xgb + 1))


#---------------------------------------------------------------------------------
# 4. Support Vector Machine (SVM)
#---------------------------------------------------------------------------------

svm_model <- svm(wl_physical ~ ., data = train_data, kernel = "radial")
svm_pred <- predict(svm_model, X_test)
confusionMatrix(svm_pred, y_test)

#---------------------------------------------------------------------------------
# 5. K-Nearest Neighbors (KNN)
#---------------------------------------------------------------------------------

# One-hot encode all predictors
X_train_dummy <- model.matrix(~ . - 1, data = X_train)
X_test_dummy  <- model.matrix(~ . - 1, data = X_test)

# Normalize
normalize <- function(x) { (x - min(x)) / (max(x) - min(x)) }
X_train_knn <- as.data.frame(apply(X_train_dummy, 2, normalize))
X_test_knn  <- as.data.frame(apply(X_test_dummy, 2, normalize))

# Run KNN
knn_pred <- knn(train = X_train_knn, test = X_test_knn, cl = y_train, k = 5)
confusionMatrix(knn_pred, y_test)



##################################################################################
#---------------------------------------------------------------------------------
# 3 Classes (1–2, 3–5, 6–7)
#---------------------------------------------------------------------------------
##################################################################################


#---------------------------------------------------------------------------------
# Data preprocessing 
#---------------------------------------------------------------------------------

collapse_to_3 <- function(x) {
  ifelse(x %in% c(1, 2), "Low",
         ifelse(x %in% c(3, 4, 5), "Medium", "High"))
}

data$wl_physical_3 <- as.factor(collapse_to_3(as.numeric(as.character(data$wl_physical))))

# Repeat similar steps as above
train_index <- createDataPartition(data$wl_physical_3, p = 0.8, list = FALSE)
train_data_3 <- data[train_index, ]
test_data_3 <- data[-train_index, ]

X_train <- train_data_3 %>% select(-wl_physical, -wl_physical_3)
y_train <- train_data_3$wl_physical_3
X_test  <- test_data_3 %>% select(-wl_physical, -wl_physical_3)
y_test  <- test_data_3$wl_physical_3

#---------------------------------------------------------------------------------
# Random Forest
#---------------------------------------------------------------------------------
rf_model_3 <- randomForest(x = X_train, y = y_train)
rf_pred_3 <- predict(rf_model_3, X_test)
confusionMatrix(rf_pred_3, y_test)



#---------------------------------------------------------------------------------
# 
#---------------------------------------------------------------------------------

#---------------------------------------------------------------------------------
# 
#---------------------------------------------------------------------------------

#---------------------------------------------------------------------------------
# 
#---------------------------------------------------------------------------------

#---------------------------------------------------------------------------------
# 
#---------------------------------------------------------------------------------

#---------------------------------------------------------------------------------
# 
#---------------------------------------------------------------------------------

#---------------------------------------------------------------------------------
# 
#---------------------------------------------------------------------------------

#---------------------------------------------------------------------------------
# 
#---------------------------------------------------------------------------------

#---------------------------------------------------------------------------------
# 
#---------------------------------------------------------------------------------

#---------------------------------------------------------------------------------
# 
#---------------------------------------------------------------------------------

#---------------------------------------------------------------------------------
# 
#---------------------------------------------------------------------------------



##################################################################################
#---------------------------------------------------------------------------------
# 2 Classes (1–2 = Low, 3–7 = High)
#---------------------------------------------------------------------------------
##################################################################################


#---------------------------------------------------------------------------------
# Data preprocessing 
#---------------------------------------------------------------------------------


collapse_to_2 <- function(x) {
  ifelse(x %in% c(1, 2), "Low", "High")
}

data$wl_physical_2 <- as.factor(collapse_to_2(as.numeric(as.character(data$wl_physical))))

# Repeat split
train_index <- createDataPartition(data$wl_physical_2, p = 0.8, list = FALSE)
train_data_2 <- data[train_index, ]
test_data_2 <- data[-train_index, ]

X_train <- train_data_2 %>% select(-wl_physical, -wl_physical_2)
y_train <- train_data_2$wl_physical_2
X_test  <- test_data_2 %>% select(-wl_physical, -wl_physical_2)
y_test  <- test_data_2$wl_physical_2

# Example: Logistic Regression
glm_model <- glm(y_train ~ ., data = cbind(X_train, y_train), family = "binomial")
glm_pred <- ifelse(predict(glm_model, newdata = X_test, type = "response") > 0.5, "High", "Low")
confusionMatrix(as.factor(glm_pred), y_test)

#---------------------------------------------------------------------------------
# Logistic Regression
#---------------------------------------------------------------------------------

# glm_model <- glm(y_train ~ ., data = cbind(X_train, y_train), family = "binomial")
# glm_pred <- ifelse(predict(glm_model, newdata = X_test, type = "response") > 0.5, "High", "Low")
# confusionMatrix(as.factor(glm_pred), y_test)

#---------------------------------------------------------------------------------
# 
#---------------------------------------------------------------------------------
# 1. Impute NAs with median
X_train_clean <- X_train %>% 
  mutate(across(everything(), ~ ifelse(is.na(.), median(., na.rm = TRUE), .)))

X_test_clean <- X_test %>% 
  mutate(across(everything(), ~ ifelse(is.na(.), median(., na.rm = TRUE), .)))

# 2. Convert character columns to factors
X_train_clean <- X_train_clean %>% 
  mutate(across(where(is.character), as.factor))

X_test_clean <- X_test_clean %>% 
  mutate(across(where(is.character), as.factor))

library(randomForest)

rf_model <- randomForest(as.factor(y_train) ~ ., 
                         data = cbind(X_train_clean, y_train), 
                         ntree = 100)

rf_pred <- predict(rf_model, newdata = X_test_clean)

confusionMatrix(as.factor(rf_pred), y_test)

#---------------------------------------------------------------------------------
# 
#---------------------------------------------------------------------------------

#---------------------------------------------------------------------------------
# 
#---------------------------------------------------------------------------------

#---------------------------------------------------------------------------------
# 
#---------------------------------------------------------------------------------

#---------------------------------------------------------------------------------
# 
#---------------------------------------------------------------------------------

#---------------------------------------------------------------------------------
# 
#---------------------------------------------------------------------------------

#---------------------------------------------------------------------------------
# 
#---------------------------------------------------------------------------------

#---------------------------------------------------------------------------------
# 





# variable to be clarified
WL_Total_100 <- data.frame(WL_Total_100 = final_data_for_analysis$WL_Total_100)

library(tibble)
confused_variables <- tibble(WL_Total_100 = final_data_for_analysis$WL_Total_100,
                       Final_RVU_Charge_Map = final_data_for_analysis$Final_RVU_Charge_Map,
                       RVU_Total = final_data_for_analysis$RVU_Total,
                       contact_patient_number = final_data_for_analysis$contact_patient_number,
                       UNIQUE_MRN_COUNT = final_data_for_analysis$UNIQUE_MRN_COUNT,
                       UAL_SECONDS_ACTIVE = final_data_for_analysis$UAL_SECONDS_ACTIVE,
                       UAL_SECONDS_ACTIVE_no_specific_patient = final_data_for_analysis$UAL_SECONDS_ACTIVE_no_specific_patient,
                       patient_contact_time = final_data_for_analysis$patient_contact_time, 
                       contact_time_no_specific_patient = final_data_for_analysis$contact_time_no_specific_patient,
                       Length_of_Patient_Stay_hrs = final_data_for_analysis$Length_of_Patient_Stay_hrs,
                       Total_Length_of_Stay = final_data_for_analysis$`Total Length of Stay`
)
