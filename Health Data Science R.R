# ============================================================
# Research Question: Is sleep duration, combined with COVID-19 
# infection status, associated with heart disease risk?
# ============================================================

# ── 1. PACKAGE INSTALLATION & LOADING ───────────────────────

# Required packages: tidyverse, caret, rpart, randomForest, pROC, corrplot, fastshap, shapviz

library(tidyverse)
library(caret)
library(rpart)
library(randomForest)
library(pROC)
library(corrplot)
library(fastshap)
library(shapviz)

# ── 2. DATA LOADING ─────────────────────────────────────────

data <- read.csv("heart_2022_with_nans.csv")


# ── 3. DATA PREPARATION ─────────────────────────────────────

# Filter to COVID-positive individuals only
# Includes both clinician-confirmed and home test positives
covid_data <- data %>%
  filter(CovidPos == "Yes" | 
           CovidPos == "Tested positive using home test without a health professional")

# Select 8 clinically relevant variables
covid_data <- covid_data %>%
  select(HadHeartAttack, SleepHours, AgeCategory, Sex, BMI,
         PhysicalActivities, MentalHealthDays, GeneralHealth)

# Check missingness
colSums(is.na(covid_data))

# SleepHours: listwise deletion
# Justification: primary predictor — imputation would introduce
# synthetic values into the variable the sub-question depends on,
# compromising analytical integrity regardless of missingness mechanism
covid_data <- covid_data %>%
  filter(!is.na(SleepHours))

# BMI and MentalHealthDays: median imputation
# Justification: covariates only — imputation is appropriate.
# Median chosen over mean due to right-skewed BMI distribution
# and zero-inflated MentalHealthDays distribution
covid_data <- covid_data %>%
  mutate(
    BMI = ifelse(is.na(BMI), median(BMI, na.rm = TRUE), BMI),
    MentalHealthDays = ifelse(is.na(MentalHealthDays),
                              median(MentalHealthDays, na.rm = TRUE),
                              MentalHealthDays)
  )

# Confirm no remaining missing values
colSums(is.na(covid_data))

# Rename outcome variable
# Note: HadHeartAttack is reframed as a broader heart disease indicator
# per Pytlak (2024) Kaggle dataset documentation
covid_data <- covid_data %>%
  rename(HadHeartDisease = HadHeartAttack)

# Convert categorical variables to factors
covid_data <- covid_data %>%
  mutate(
    HadHeartDisease    = as.factor(HadHeartDisease),
    Sex                = as.factor(Sex),
    AgeCategory        = as.factor(AgeCategory),
    GeneralHealth      = as.factor(GeneralHealth),
    PhysicalActivities = as.factor(PhysicalActivities)
  )

# Remove rows with empty string levels
covid_data <- covid_data %>%
  filter(HadHeartDisease    != "" &
           AgeCategory        != "" &
           PhysicalActivities != "" &
           GeneralHealth      != "")

# Drop unused factor levels
covid_data <- droplevels(covid_data)

# Confirm final class distribution
table(covid_data$HadHeartDisease)
# Expected: ~114,866 No vs ~5,760 Yes (95/5 split)


# ── 4. MULTICOLLINEARITY CHECK ──────────────────────────────

# Check correlations among continuous predictors only
continuous_vars <- covid_data %>%
  select(SleepHours, BMI, MentalHealthDays)

cor_matrix <- cor(continuous_vars, use = "complete.obs")
print(cor_matrix)

corrplot(cor_matrix, method = "color", type = "upper",
         tl.cex = 0.8, title = "Correlation Heatmap",
         mar = c(0, 0, 1, 0))
# All correlations close to zero — no multicollinearity detected


# ── 5. TRAIN / TEST SPLIT ───────────────────────────────────

set.seed(42)

train_index <- createDataPartition(covid_data$HadHeartDisease,
                                   p = 0.8, list = FALSE)

train_data <- covid_data[train_index, ]
test_data  <- covid_data[-train_index, ]

# Confirm stratification preserved class balance
prop.table(table(train_data$HadHeartDisease))
prop.table(table(test_data$HadHeartDisease))


# ── 6. CROSS-VALIDATION CONTROL ─────────────────────────────

# Downsampling applied within each training fold only
# Test set kept at natural 95/5 distribution to reflect
# real-world conditions
ctrl <- trainControl(
  method          = "cv",
  number          = 5,
  classProbs      = TRUE,
  summaryFunction = twoClassSummary,
  sampling        = "down"
)


# ── 7. MODEL TRAINING ───────────────────────────────────────

# Logistic Regression — interpretable clinical baseline
model_lr <- train(
  HadHeartDisease ~ .,
  data      = train_data,
  method    = "glm",
  family    = "binomial",
  trControl = ctrl,
  metric    = "ROC"
)
print(model_lr)

# Decision Tree — non-linear rule-based comparison
# cp tuned automatically via caret grid search
model_dt <- train(
  HadHeartDisease ~ .,
  data      = train_data,
  method    = "rpart",
  trControl = ctrl,
  metric    = "ROC"
)
print(model_dt)

# Random Forest — ensemble complexity comparison
# mtry tuned across 3 values; ntree=500 selected after
# confirming negligible AUC difference vs ntree=100
model_rf <- train(
  HadHeartDisease ~ .,
  data      = train_data,
  method    = "rf",
  trControl = ctrl,
  metric    = "ROC",
  ntree     = 500
)
print(model_rf)


# ── 8. MODEL EVALUATION ─────────────────────────────────────

# Predictions on held-out test set
pred_lr <- predict(model_lr, test_data)
pred_dt <- predict(model_dt, test_data)
pred_rf <- predict(model_rf, test_data)

# Confusion matrices
# Sensitivity and AUC prioritised over accuracy given class imbalance
confusionMatrix(pred_lr, test_data$HadHeartDisease, positive = "Yes")
confusionMatrix(pred_dt, test_data$HadHeartDisease, positive = "Yes")
confusionMatrix(pred_rf, test_data$HadHeartDisease, positive = "Yes")

# Predicted probabilities for ROC/AUC calculation
prob_lr <- predict(model_lr, test_data, type = "prob")[, "Yes"]
prob_dt <- predict(model_dt, test_data, type = "prob")[, "Yes"]
prob_rf <- predict(model_rf, test_data, type = "prob")[, "Yes"]

# ROC curves and AUC
roc_lr <- roc(test_data$HadHeartDisease, prob_lr)
roc_dt <- roc(test_data$HadHeartDisease, prob_dt)
roc_rf <- roc(test_data$HadHeartDisease, prob_rf)

cat("Logistic Regression AUC:", auc(roc_lr), "\n")
cat("Decision Tree AUC:",       auc(roc_dt), "\n")
cat("Random Forest AUC:",       auc(roc_rf), "\n")

# ROC curve plot
plot(roc_lr, col = "blue", main = "ROC Curve Comparison")
plot(roc_dt, col = "red",  add = TRUE)
plot(roc_rf, col = "green",add = TRUE)
legend("bottomright",
       legend = c(
         paste("Logistic Regression (AUC =", round(auc(roc_lr), 3), ")"),
         paste("Decision Tree (AUC =",       round(auc(roc_dt), 3), ")"),
         paste("Random Forest (AUC =",       round(auc(roc_rf), 3), ")")
       ),
       col = c("blue", "red", "green"),
       lwd = 2)

# Save ROC plot
ggsave_base <- function(filename, plot_fn, width = 800, height = 600) {
  png(filename, width = width, height = height)
  plot_fn()
  dev.off()
}

png("ROC_Comparison.png", width = 800, height = 600)
plot(roc_lr, col = "blue", main = "ROC Curve Comparison")
plot(roc_dt, col = "red",  add = TRUE)
plot(roc_rf, col = "green",add = TRUE)
legend("bottomright",
       legend = c(
         paste("Logistic Regression (AUC =", round(auc(roc_lr), 3), ")"),
         paste("Decision Tree (AUC =",       round(auc(roc_dt), 3), ")"),
         paste("Random Forest (AUC =",       round(auc(roc_rf), 3), ")")
       ),
       col = c("blue", "red", "green"),
       lwd = 2)
dev.off()


# ── 9. SHAP ANALYSIS ────────────────────────────────────────

# SHAP computed on logistic regression (best performing model)
# as standard practice for the primary analysis.
# Extension to DT and RF noted as a future direction —
# would allow direct comparison of feature importance
# stability across model architectures.

# Prediction wrapper
predict_lr <- function(object, newdata) {
  predict(object, newdata = newdata, type = "prob")[, "Yes"]
}

# Test set features (outcome excluded)
X_test <- test_data[, -which(names(test_data) == "HadHeartDisease")]

# Compute SHAP values
set.seed(42)
shap_lr <- explain(
  object       = model_lr,
  X            = X_test,
  pred_wrapper = predict_lr,
  nsim         = 50
)

# Beeswarm plot
shap_viz_lr <- shapviz(as.matrix(shap_lr), X = X_test)
sv_importance(shap_viz_lr, kind = "beeswarm") +
  ggtitle("SHAP Feature Importance — Logistic Regression")

# Save beeswarm plot
beeswarm_plot <- sv_importance(shap_viz_lr, kind = "beeswarm") +
  ggtitle("SHAP Feature Importance — Logistic Regression")

ggsave("SHAP_Beeswarm_LR.png",
       plot   = beeswarm_plot,
       width  = 10,
       height = 6,
       dpi    = 300)

# Mean absolute SHAP importance table
shap_importance <- as.data.frame(shap_lr) %>%
  summarise(across(everything(), ~ mean(abs(.)))) %>%
  pivot_longer(everything(),
               names_to  = "Feature",
               values_to = "MeanAbsSHAP") %>%
  arrange(desc(MeanAbsSHAP))

print(shap_importance)


# ── 10. LIMITATIONS NOTE ────────────────────────────────────

# The following extensions were identified during analysis
# but not implemented due to scope constraints:
#
# 1. SHAP comparison across DT and RF models
#    — Would test whether SleepHours feature importance
#      ranking is stable across model architectures
#
# 2. Confounder analysis at extreme SleepHours values
#    — Re-running SHAP on subsets of patients at the
#      extreme ends of the sleep distribution (e.g. <=4hrs
#      and >=10hrs) would test whether the directional
#      finding holds at clinically meaningful thresholds
#      or whether confounders such as post-COVID fatigue
#      drive the association at extremes
#
# 3. Polynomial extension
#    — Adding SleepHours^2 to the logistic regression
#      would directly test the U-shaped hypothesis rather
#      than relying on indirect evidence from the
#      tree-based model comparison

# ============================================================
# END OF SCRIPT
# ============================================================