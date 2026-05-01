# UK Road Safety — Machine Learning
# Requires: df_cleaned.rds  (run cleaning.R first)
# Output  : ml_results_all.csv, b_bagging_results.csv, plots (.png)

# ── Packages ──────────────────────────────────────────────────────────────────
 
 
suppressPackageStartupMessages({
  library(dplyr); library(tidyr)
  library(ggplot2); library(gridExtra); library(scales); library(viridis)
  library(caret); library(randomForest); library(xgboost)
  library(lightgbm); library(ipred); library(ROSE); library(pROC)
})
set.seed(42)


# ── Load Cleaned Data ─────────────────────────────────────────────────────────
if (file.exists("df_cleaned.rds")) {
  df <- readRDS("df_cleaned.rds")
} else if (file.exists("df_cleaned.csv")) {
  df <- read.csv("df_cleaned.csv", stringsAsFactors = FALSE)
} else {
  stop("df_cleaned.rds not found — run cleaning.R first.")
}

# Safety net: recreate target if it arrived as all-NA
if (!"accident_seriousness" %in% names(df) || all(is.na(df$accident_seriousness)))
  df$accident_seriousness <- ifelse(df$accident_severity == 3, "Not Serious", "Serious")


# ── Build ML Feature Matrix ───────────────────────────────────────────────────
df_ml <- df %>%
  select(-any_of(c("date", "time", "accident_severity"))) %>%
  mutate(across(where(is.character), ~ as.integer(as.factor(.)) - 1L))

stopifnot("accident_seriousness" %in% names(df_ml))

X <- df_ml %>% select(-accident_seriousness)
y <- df_ml$accident_seriousness


# ── Accident Seriousness Bar Plot ─────────────────────────────────────────────
p_acc <- ggplot(df %>% count(accident_seriousness),
                aes(x = accident_seriousness, y = n, fill = accident_seriousness)) +
  geom_col(width = 0.55) +
  scale_fill_manual(values = c("Not Serious" = "#9b59b6", "Serious" = "#e74c3c")) +
  scale_y_continuous(labels = comma) +
  labs(title = "Accident Seriousness", x = "", y = "Number of Accidents") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "none",
        plot.title = element_text(face = "bold", size = 16))
ggsave("accident_seriousness.png", p_acc, width = 8, height = 5)


# ── Train / Test Split (75 / 25) ──────────────────────────────────────────────
train_idx <- createDataPartition(y, p = 0.75, list = FALSE)
X_train <- X[ train_idx, ];  X_test <- X[-train_idx, ]
y_train <- y[ train_idx];    y_test <- y[-train_idx]


# ── Undersample Majority for Method 1 ────────────────────────────────────────
serious     <- data.frame(X_train, accident_seriousness = y_train) %>% filter(accident_seriousness == 1)
not_serious <- data.frame(X_train, accident_seriousness = y_train) %>% filter(accident_seriousness == 0)
train_bal   <- bind_rows(serious,
                          not_serious %>% slice_sample(n = nrow(serious), replace = TRUE)) %>%
               slice_sample(prop = 1)

X_bal      <- train_bal %>% select(-accident_seriousness)
y_bal      <- train_bal$accident_seriousness
X_bal_mat  <- as.matrix(X_bal)
X_test_mat <- as.matrix(X_test)


# ── Helper: Confusion Matrix Heatmap ─────────────────────────────────────────
cm_plot <- function(y_true, y_pred, model_name) {
  cm    <- table(Actual    = factor(y_true, levels = c(0,1)),
                 Predicted = factor(y_pred, levels = c(0,1)))
  df_cm <- as.data.frame(cm)
  names(df_cm) <- c("Actual","Predicted","Freq")
  df_cm$label  <- paste0(
    dplyr::case_when(
      df_cm$Actual=="1" & df_cm$Predicted=="1" ~ "TP",
      df_cm$Actual=="0" & df_cm$Predicted=="0" ~ "TN",
      df_cm$Actual=="1" & df_cm$Predicted=="0" ~ "FN",
      TRUE                                     ~ "FP"),
    "\n", df_cm$Freq)
  ggplot(df_cm, aes(x = Predicted, y = Actual, fill = Freq)) +
    geom_tile(colour = "white", linewidth = 1) +
    geom_text(aes(label = label), size = 5.5, fontface = "bold") +
    scale_fill_gradient(low = "#deebf7", high = "#08519c") +
    labs(title = paste("Confusion Matrix —", model_name),
         x = "Predicted", y = "Actual") +
    theme_minimal(base_size = 13) +
    theme(legend.position = "none",
          plot.title = element_text(face = "bold"))
}


# ── Helper: Compute Metrics ───────────────────────────────────────────────────
compute_metrics <- function(name, y_true, y_pred, y_prob, cv_acc) {
  cm   <- table(Actual    = factor(y_true, levels = c(0,1)),
                Predicted = factor(y_pred, levels = c(0,1)))
  tn <- cm[1,1]; fp <- cm[1,2]; fn <- cm[2,1]; tp <- cm[2,2]
  acc  <- (tp+tn)/sum(cm)*100
  rec  <- ifelse((tp+fn)>0, tp/(tp+fn)*100, 0)
  prec <- ifelse((tp+fp)>0, tp/(tp+fp)*100, 0)
  f1   <- ifelse((prec+rec)>0, 2*prec*rec/(prec+rec), 0)
  fpr  <- ifelse((tn+fp)>0,   fp/(tn+fp)*100, 0)
  roc_v <- tryCatch(
    as.numeric(pROC::auc(pROC::roc(y_true, y_prob, quiet=TRUE)))*100,
    error = function(e) NA_real_)
  ll <- -mean(y_true*log(y_prob+1e-15)+(1-y_true)*log(1-y_prob+1e-15))
  data.frame(Classifier  = name,
             Accuracy    = round(acc,   2),
             Log_Loss    = round(ll,    3),
             Cross_Val   = round(cv_acc*100, 2),
             Recall      = round(rec,   2),
             Roc_Auc     = round(roc_v, 2),
             F1          = round(f1,    2),
             FPR         = round(fpr,   2),
             Error_Rate  = round(100-acc, 2),
             stringsAsFactors = FALSE)
}


# ── Helper: Fit caret model, print CM, return metrics ────────────────────────
run_caret <- function(label, fit, X_te, y_te) {
  prob <- predict(fit, X_te, type = "prob")[, "S"]
  pred <- as.integer(prob > 0.5)
  cv   <- max(fit$results$Accuracy)
  print(cm_plot(y_te, pred, label))
  cat(sprintf("  %-38s Accuracy: %5.2f%%\n", label, mean(pred == y_te)*100))
  compute_metrics(label, y_te, pred, prob, cv)
}


# ── METHOD 1 — Standard Classifiers ──────────────────────────────────────────
cat("\n========================================\n")
cat("  METHOD 1 — Standard Classifiers\n")
cat("========================================\n")

results     <- data.frame()
ctrl3       <- trainControl(method="cv", number=3, verboseIter=FALSE, classProbs=TRUE)
train_caret <- data.frame(X_bal, seriousness = factor(y_bal, labels=c("NS","S")))

cat("  [1/5] Bagging ...\n")
results <- bind_rows(results, run_caret("BaggingClassifier",
  train(seriousness~., data=train_caret, method="treebag", trControl=ctrl3),
  X_test, y_test))

cat("  [2/5] AdaBoost ...\n")
results <- bind_rows(results, run_caret("AdaBoostClassifier",
  train(seriousness~., data=train_caret, method="ada", trControl=ctrl3,
        tuneGrid=expand.grid(iter=100, maxdepth=4, nu=0.05)),
  X_test, y_test))

cat("  [3/5] Random Forest ...\n")
results <- bind_rows(results, run_caret("RandomForestClassifier",
  train(seriousness~., data=train_caret, method="rf", ntree=200, trControl=ctrl3,
        tuneGrid=data.frame(mtry=floor(sqrt(ncol(X_bal))))),
  X_test, y_test))

cat("  [4/5] LightGBM ...\n")
lgb_p    <- list(objective="binary", metric="binary_logloss", learning_rate=0.03,
                 max_depth=10, num_leaves=50, min_data_in_leaf=10, verbose=-1)
lgb_mod  <- lgb.train(lgb_p, lgb.Dataset(X_bal_mat, label=y_bal), nrounds=200, verbose=-1)
lgb_prob <- predict(lgb_mod, X_test_mat)
lgb_pred <- as.integer(lgb_prob > 0.5)
lgb_cv   <- mean(sapply(createFolds(y_bal, k=3, list=TRUE), function(idx) {
  m <- lgb.train(lgb_p, lgb.Dataset(X_bal_mat[-idx,], label=y_bal[-idx]), nrounds=100, verbose=-1)
  mean(as.integer(predict(m, X_bal_mat[idx,])>0.5)==y_bal[idx])
}))
print(cm_plot(y_test, lgb_pred, "LightGBM"))
cat(sprintf("  %-38s Accuracy: %5.2f%%\n", "LGBMClassifier", mean(lgb_pred==y_test)*100))
results <- bind_rows(results, compute_metrics("LGBMClassifier", y_test, lgb_pred, lgb_prob, lgb_cv))

cat("  [5/5] XGBoost ...\n")
xgb_tr   <- xgb.DMatrix(X_bal_mat,  label=y_bal)
xgb_te   <- xgb.DMatrix(X_test_mat, label=y_test)
xgb_par  <- list(objective="binary:logistic", eval_metric="logloss",
                 eta=0.05, max_depth=8, gamma=1, subsample=1, verbosity=0)
xgb_mod  <- xgb.train(xgb_par, xgb_tr, nrounds=200)
xgb_prob <- predict(xgb_mod, xgb_te)
xgb_pred <- as.integer(xgb_prob > 0.5)
xgb_cv   <- 1 - min(xgb.cv(xgb_par, xgb_tr, nrounds=100, nfold=3,
                             verbose=FALSE)$evaluation_log$test_logloss_mean)
print(cm_plot(y_test, xgb_pred, "XGBoost"))
cat(sprintf("  %-38s Accuracy: %5.2f%%\n", "XGBClassifier", mean(xgb_pred==y_test)*100))
results <- bind_rows(results, compute_metrics("XGBClassifier", y_test, xgb_pred, xgb_prob, xgb_cv))


# ── METHOD 2 — Balanced Classifiers ──────────────────────────────────────────
cat("\n========================================\n")
cat("  METHOD 2 — Balanced Classifiers\n")
cat("========================================\n")

results2         <- data.frame()
full_train_caret <- data.frame(X_train, seriousness=factor(y_train, labels=c("NS","S")))
rose_df          <- ROSE(seriousness~., data=full_train_caret,
                         seed=42, N=nrow(full_train_caret))$data

cat("  [1/3] Balanced Bagging ...\n")
bbag_fit <- train(seriousness~., data=rose_df, method="treebag", trControl=ctrl3)
results2 <- bind_rows(results2, run_caret("BalancedBaggingClassifier", bbag_fit, X_test, y_test))
bbag_cv  <- max(bbag_fit$results$Accuracy)

cat("  [2/3] Easy Ensemble (5 bags) ...\n")
ee_probs <- matrix(NA, nrow=nrow(X_test), ncol=5)
for (i in 1:5) {
  ri         <- ROSE(seriousness~., data=full_train_caret, seed=i, N=nrow(full_train_caret))$data
  mi         <- train(seriousness~., data=ri, method="treebag",
                      trControl=trainControl(method="none"))
  ee_probs[,i] <- predict(mi, X_test, type="prob")[,"S"]
}
ee_prob  <- rowMeans(ee_probs)
ee_pred  <- as.integer(ee_prob > 0.5)
print(cm_plot(y_test, ee_pred, "Easy Ensemble"))
cat(sprintf("  %-38s Accuracy: %5.2f%%\n", "EasyEnsembleClassifier", mean(ee_pred==y_test)*100))
results2 <- bind_rows(results2, compute_metrics("EasyEnsembleClassifier", y_test, ee_pred, ee_prob, bbag_cv))

cat("  [3/3] Balanced Random Forest ...\n")
results2 <- bind_rows(results2, run_caret("BalancedRandomForestClassifier",
  train(seriousness~., data=rose_df, method="rf", ntree=200, trControl=ctrl3,
        tuneGrid=data.frame(mtry=floor(sqrt(ncol(X_train))))),
  X_test, y_test))


# ── Combined Results ──────────────────────────────────────────────────────────
ml_results <- bind_rows(results, results2)

cat("\n========================================\n")
cat("  ALL MODEL RESULTS\n")
cat("========================================\n")
print(ml_results, row.names=FALSE)
write.csv(ml_results, "ml_results_all.csv", row.names=FALSE)

metric_cols <- c("Accuracy","Log_Loss","Cross_Val","Recall","Roc_Auc","F1","FPR","Error_Rate")
long_res    <- ml_results %>%
  tidyr::pivot_longer(cols=all_of(metric_cols), names_to="Metric", values_to="Value")
plot_list   <- lapply(metric_cols, function(m) {
  ggplot(dplyr::filter(long_res, Metric==m),
         aes(x=Value, y=reorder(Classifier,Value), fill=Classifier)) +
    geom_col(show.legend=FALSE) +
    scale_fill_viridis_d(option="plasma") +
    labs(title=gsub("_"," ",m), x=NULL, y=NULL) +
    theme_minimal(base_size=9) +
    theme(plot.title=element_text(face="bold", size=9))
})
ggsave("ml_scores_all_models.png",
       do.call(grid.arrange, c(plot_list, ncol=2)), width=14, height=18)


# ── Best Model: Balanced Bagging + LightGBM ───────────────────────────────────
cat("\n========================================\n")
cat("  BEST MODEL: Balanced Bagging + LightGBM\n")
cat("========================================\n")

t0         <- proc.time()
n_bags     <- 10   # increase to 500 for production
lgbm_probs <- matrix(NA, nrow=nrow(X_test), ncol=n_bags)
lgb_bp     <- list(objective="binary", metric="binary_logloss", learning_rate=0.03,
                   max_depth=10, num_leaves=50, min_data_in_leaf=10, verbose=-1)

for (b in seq_len(n_bags)) {
  rb           <- ROSE(seriousness~., data=full_train_caret, seed=b, N=nrow(full_train_caret))$data
  Xb           <- as.matrix(rb %>% select(-seriousness))
  yb           <- as.integer(rb$seriousness == "S")
  lgbm_probs[,b] <- predict(
    lgb.train(lgb_bp, lgb.Dataset(Xb, label=yb), nrounds=200, verbose=-1),
    X_test_mat)
  cat(sprintf("  Bag %02d / %02d\n", b, n_bags))
}

best_prob <- rowMeans(lgbm_probs)
best_pred <- as.integer(best_prob > 0.5)
elapsed   <- (proc.time() - t0)[["elapsed"]]

best_cm <- table(Actual    = factor(y_test,   levels=c(0,1)),
                 Predicted = factor(best_pred, levels=c(0,1)))
tn_b <- best_cm[1,1]; fp_b <- best_cm[1,2]
fn_b <- best_cm[2,1]; tp_b <- best_cm[2,2]
acc_b  <- (tp_b+tn_b)/sum(best_cm)*100
spec_b <- tn_b/(tn_b+fp_b)*100
fpr_b  <- fp_b/(tn_b+fp_b)*100
f1_b   <- 2*tp_b/(2*tp_b+fp_b+fn_b)*100
prec_b <- tp_b/(tp_b+fp_b)*100
rec_b  <- tp_b/(tp_b+fn_b)*100
roc_b  <- as.numeric(pROC::auc(pROC::roc(y_test, best_prob, quiet=TRUE)))*100

cm_df <- as.data.frame(best_cm); names(cm_df) <- c("Actual","Predicted","Freq")
p_cm  <- ggplot(cm_df, aes(x=Predicted, y=Actual, fill=Freq)) +
  geom_tile(colour="black", linewidth=0.5) +
  geom_text(aes(label=Freq), size=9, fontface="bold") +
  scale_fill_viridis_c(option="viridis") +
  labs(title=sprintf("Balanced Bagging + LightGBM  |  Accuracy: %.2f%%", acc_b),
       x="Predicted", y="Actual") +
  theme_minimal(base_size=14) +
  theme(legend.position="none", plot.title=element_text(face="bold"))
print(p_cm)
ggsave("bbag_lgbm_confusion_matrix.png", p_cm, width=7, height=5)

cat(sprintf("\n  %-22s %6.2f%%\n", "Accuracy:",       acc_b))
cat(sprintf("  %-22s %6.2f%%\n",  "F1 Score:",        f1_b))
cat(sprintf("  %-22s %6.2f%%\n",  "Precision:",       prec_b))
cat(sprintf("  %-22s %6.2f%%\n",  "Recall:",          rec_b))
cat(sprintf("  %-22s %6.2f%%\n",  "ROC AUC:",         roc_b))
cat(sprintf("  %-22s %6.2f%%\n",  "Specificity:",     spec_b))
cat(sprintf("  %-22s %6.2f%%\n",  "False Pos Rate:",  fpr_b))
cat(sprintf("  %-22s %6.2f%%\n",  "Error Rate:",      100-acc_b))
cat(sprintf("  %-22s %6.2f min\n","Time:",             elapsed/60))


# ── Final Comparison Table ────────────────────────────────────────────────────
b_bagging_summary <- data.frame(
  Algorithm   = c("Balanced Bagging", "Balanced Bagging + LightGBM"),
  Accuracy    = c(round(bbag_cv*100, 2), round(acc_b,  2)),
  F1          = c(NA,                    round(f1_b,   2)),
  Precision   = c(NA,                    round(prec_b, 2)),
  Recall      = c(NA,                    round(rec_b,  2)),
  Specificity = c(NA,                    round(spec_b, 2)),
  Error_Rate  = c(NA,                    round(100-acc_b, 2)),
  FPR         = c(NA,                    round(fpr_b,  2)),
  ROC_AUC     = c(NA,                    round(roc_b,  2)),
  Time_min    = c(NA,                    round(elapsed/60, 2)),
  Library     = c("ROSE + ipred",        "ROSE + LightGBM"),
  stringsAsFactors = FALSE
)

cat("\n========================================\n")
cat("  BALANCED BAGGING COMPARISON\n")
cat("========================================\n")
print(b_bagging_summary, row.names=FALSE)
write.csv(b_bagging_summary, "b_bagging_results.csv", row.names=FALSE)

cat("\n  Saved:\n")
cat("    accident_seriousness.png\n")
cat("    ml_scores_all_models.png\n")
cat("    bbag_lgbm_confusion_matrix.png\n")
cat("    ml_results_all.csv\n")
cat("    b_bagging_results.csv\n")
cat("========================================\n")