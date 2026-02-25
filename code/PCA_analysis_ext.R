# ---------------------------------------------------------------
# Project: PCA on Dependent & Independent Variables
# Author: Md Mohiuddin Adnan
# Date: 2025-08-13
# Description: Perform PCA separately on workload outcomes 
#              (dependent vars) and clinical/operational features (predictors).
# ---------------------------------------------------------------

# =============================================
# Workload PCA & Score Computation
# =============================================

# libraries
library(psych)       # For PCA, parallel analysis
library(GPArotation) # For rotation options
library(janitor)     
library(readxl)
library(tidyverse)

# -----------------------------
# 0. import and clean data 
# -----------------------------
# load data
PCA_data <- read_tsv("data/ModelingData.txt")

# clean column names
PCA_data <- janitor::clean_names(PCA_data)

dependent_vars <- c(
  "physical", "mental", "temporal",
  "performance", "effort", "frustration",
  "time", "chart"
)

dep_data <- PCA_data %>% select(all_of(dependent_vars))

# -----------------------------
# 1. Check correlation matrix
# -----------------------------
cor_matrix <- cor(dep_data, use = "pairwise.complete.obs")
print(cor_matrix)

# -----------------------------
# 2. Parallel analysis to decide number of components
# -----------------------------
png("parallel_plot_pca.png", width = 800, height = 600)
fa.parallel(dep_data, fa = "pc")
pa
dev.off()

# -----------------------------
# 3. Standardize the data
# -----------------------------
dep_data_scaled <- scale(dep_data)

# -----------------------------
# 4. Run PCA
# -----------------------------
# Suppose parallel analysis suggests 3 components
pca_result <- principal(dep_data_scaled, nfactors = 3, rotate = "varimax")

# -----------------------------
# 5. Inspect PCA loadings
# -----------------------------
print(pca_result, cut = 0.3)   # suppress loadings <0.3
pca_result$loadings            # full loadings

# -----------------------------
# 6. Extract PCA scores
# -----------------------------
pca_scores <- as.data.frame(pca_result$scores)
head(pca_scores)

# -----------------------------
# 7. Compute overall workload score
# -----------------------------
# Option A: simple sum of component scores
workload_score <- rowSums(pca_scores)

# Option B: weighted by variance explained
weights <- pca_result$Vaccounted["Proportion Var", ]
workload_score_weighted <- as.matrix(pca_scores) %*% weights

# Make scores all positive (optional, for allocation purposes)
workload_score_pos <- workload_score_weighted - min(workload_score_weighted)

# Inspect final workload scores
head(workload_score_pos)

# -----------------------------
# 8. Next steps
# -----------------------------
# Use workload_score_pos to allocate workload across physicians:
# - Sum current workload per physician
# - Assign new patients to the physician with least workload


# # Load packages
# library(janitor)
# library(readxl)
# library(tidyverse)
# 
# 
# 
#---------------------------------------------------------------
# 1. load dataset
#---------------------------------------------------------------
# load data
PCA_data <- read_excel("data/final_data_for_analysis.xlsx")
# Tab-delimited
PCA_data_2 <- read_tsv("data/ModelingData.txt")

# clean column names
PCA_data <- janitor::clean_names(PCA_data)
PCA_data_2 <- janitor::clean_names(PCA_data_2)

dependent_vars <- c(
  "wl_physical", "wl_mental", "wl_temporal",
  "wl_performance", "wl_effort", "wl_frustration","los")

dependent_vars_2 <- c(
  "physical", "mental", "temporal",
  "performance", "effort", "frustration"
  #,"time", "chart"
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

# Clean and select
PCA_data_clean <- PCA_data_2 %>%
  rename(
    los = length_of_patient_stay_hrs,
    CCIS = charlson_comorbidity_index_score_column,
    ACH_score = ach_eligibility_v3_score
  ) %>%
  select(all_of(c(dependent_vars, independent_vars))) %>%
  drop_na()

dep_data <- PCA_data_clean %>% select(all_of(dependent_vars))
dep_data_2 <- PCA_data_2 %>% select(all_of(dependent_vars_2))

ind_data <- PCA_data_2 %>% select(all_of(independent_vars)) %>% drop_na()
write.csv(ind_data, "data/ind_data.csv", row.names = FALSE)
#---------------------------------------------------------------
# 3. PCA on dependent variables (workload outcomes)
#---------------------------------------------------------------

# Version B: Collapse Likert to 0-2, keep LOS as is
PCA_data_collapsed <- PCA_data %>%
  mutate(across(all_of(setdiff(dependent_vars_2, c("time", "chart"))), ~ case_when(
    . == 1  ~ 0,
    . >= 2 & . <= 6 ~ 1,
    . == 7 ~ 2,
    TRUE ~ NA_real_
  )))

# PCA with scaling
pca_dep <- prcomp(dep_data_2, scale. = TRUE)
summary(pca_dep)
pca_dep$rotation
plot(pca_dep, type = "l", main = "Scree Plot - Dependent Variables")


#---------------------------------------------------------------
# 4. PCA on independent variables (predictors)
#---------------------------------------------------------------
# Create dummy variables (one-hot encoding)
ind_data_dummy <- dummyVars(" ~ .", data = ind_data) %>%
  predict(newdata = ind_data) %>%
  as.data.frame()

pca_ind <- prcomp(ind_data_dummy, scale. = TRUE)
summary(pca_ind)       # Variance explained
pca_ind$rotation       # Loadings
# Scree plot
plot(pca_ind, type = "l", main = "Scree Plot - Independent Variables")

#---------------------------------------------------------------
# 5. Get PCA scores if you want to use them in modeling
#---------------------------------------------------------------
dep_scores <- as.data.frame(pca_dep$x)   # PC1, PC2, ...
ind_scores <- as.data.frame(pca_ind$x)   # PC1, PC2, ...

###########################
# scatter plots of PCs
###########################
dep_scores_physical <- dep_scores %>%
  mutate(physical = dep_data_2$physical)

library(ggplot2)
library(plotly)

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
# plot for all dependent variables
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
      title = paste("PC1 vs PC2 colored by", dep_var),
      x = "PC1",
      y = "PC2",
      color = dep_var
    )
  
  # Convert to interactive Plotly
  interactive_plot <- ggplotly(p, tooltip = "text")
  
  # Save as HTML
  htmlwidgets::saveWidget(interactive_plot, paste0("reports/",dep_var, "_PC1_PC2_scatter.html"))
}

#-------------------------------------------------------
# Optional: keep first N PCs (example: first 2 PCs from each set)
dep_scores_reduced <- dep_scores %>% select(PC1, PC2)
ind_scores_reduced <- ind_scores %>% select(PC1, PC2)

#---------------------------------------------------------------
# 6. Example: Model first dep PC using first 2 ind PCs
#---------------------------------------------------------------
lm_model <- lm(PC1 ~ PC1 + PC2,
               data = cbind(dep_scores_reduced, ind_scores_reduced))
summary(lm_model)
