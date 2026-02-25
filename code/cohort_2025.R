

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

length(unique(Cohort_2025$pat_enc_csn_id))

missing_rvu <- Cohort_2025 %>%
  group_by(pat_enc_csn_id, hosp_date) %>%
  summarise(
    missing_rvu_count = sum(is.na(rvu_total)),
    total_rows = n(),
    .groups = "drop"
  )

missing_rvu