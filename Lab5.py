
# ============================================================
# LAB 5: K-Nearest Neighbor (KNN) for Classification and Regression
# Dataset: Pima Indians Diabetes (diabetes.csv)
# Student: Yash Kapse
# ============================================================

# 1. Import required libraries
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
    KFold,
    cross_val_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

np.random.seed(42)
sns.set_theme(style="whitegrid")


# ============================================================
# PART A: KNN CLASSIFICATION
# Predict diabetes Outcome
# ============================================================

# 2. Load and explore dataset
df = pd.read_csv("diabetes.csv")

print("\nFirst five rows:")
print(df.head())

print("\nLast five rows:")
print(df.tail())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nDataset information:")
df.info()

print("\nStatistical summary:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

# 3. Remove duplicates and missing target values
df = df.drop_duplicates().copy()
df = df.dropna(subset=["Outcome"]).copy()

# Ensure Outcome is binary
df["Outcome"] = pd.to_numeric(df["Outcome"], errors="coerce")
df = df.dropna(subset=["Outcome"]).copy()
df["Outcome"] = df["Outcome"].astype(int)

if not set(df["Outcome"].unique()).issubset({0, 1}):
    raise ValueError("Outcome must contain only 0 and 1.")

# 4. Correlation heatmap
numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(10, 7))
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5
)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

# 5. Identify invalid zero values
zero_columns = [
    col for col in
    ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    if col in df.columns
]

print("\nInvalid zero counts:")
print((df[zero_columns] == 0).sum())

# 6. Prepare features and target
X = df.drop(columns=["Outcome"]).apply(
    pd.to_numeric, errors="coerce"
)
y = df["Outcome"]

# Replace invalid zeros with NaN in selected clinical features
X = X.copy()
X[zero_columns] = X[zero_columns].replace(0, np.nan)

print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)

print("\nClass distribution:")
print(y.value_counts().sort_index())

# 7. Train/test split (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

print("\nTraining class distribution:")
print(y_train.value_counts(normalize=True))

print("\nTesting class distribution:")
print(y_test.value_counts(normalize=True))

# 8. Build preprocessing and KNN pipeline
# Imputation and scaling are fitted only on training folds.
knn_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5))
])

# 9. Train KNN classifier (K = 5)
knn_pipeline.fit(X_train, y_train)

# 10. Predictions
y_pred = knn_pipeline.predict(X_test)

print("\nFirst 20 predictions:")
print(y_pred[:20])

# 11. Classification evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(
    y_test, y_pred, zero_division=0
)
recall = recall_score(
    y_test, y_pred, zero_division=0
)
f1 = f1_score(
    y_test, y_pred, zero_division=0
)

print("\nKNN Classification Metrics")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["No Diabetes", "Diabetes"],
    zero_division=0
))

# 12. Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(7, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    xticklabels=["No Diabetes", "Diabetes"],
    yticklabels=["No Diabetes", "Diabetes"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("KNN Classification Confusion Matrix")
plt.tight_layout()
plt.show()

# 13. Effect of K on training and testing accuracy
k_values = range(1, 31)
train_accuracy = []
test_accuracy = []

for k in k_values:
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k))
    ])

    model.fit(X_train, y_train)

    train_accuracy.append(model.score(X_train, y_train))
    test_accuracy.append(model.score(X_test, y_test))

plt.figure(figsize=(10, 6))
plt.plot(
    k_values, train_accuracy,
    marker="o", label="Training Accuracy"
)
plt.plot(
    k_values, test_accuracy,
    marker="o", label="Testing Accuracy"
)
plt.xlabel("Number of Neighbors (K)")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy vs K")
plt.xticks(list(k_values))
plt.legend()
plt.tight_layout()
plt.show()

# Test accuracy is displayed for analysis only.
# Do not use it to select the final model.
print("\nK values and test accuracies:")
for k, score in zip(k_values, test_accuracy):
    print(f"K={k:2d}: {score:.4f}")

# 14. Cross-validation to select K
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = []

for k in k_values:
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k))
    ])

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1
    )

    cv_scores.append(scores.mean())

plt.figure(figsize=(10, 6))
plt.plot(list(k_values), cv_scores, marker="o")
plt.xlabel("Number of Neighbors (K)")
plt.ylabel("Mean Cross-validation Accuracy")
plt.title("KNN: K vs Cross-validation Accuracy")
plt.xticks(list(k_values))
plt.tight_layout()
plt.show()

best_k_cv = list(k_values)[int(np.argmax(cv_scores))]

print("\nBest K from cross-validation:", best_k_cv)
print("Best CV accuracy:", max(cv_scores))

# 15. Hyperparameter tuning using GridSearchCV
param_grid = {
    "knn__n_neighbors": list(range(3, 31, 2)),
    "knn__weights": ["uniform", "distance"],
    "knn__metric": ["euclidean", "manhattan", "minkowski"],
    "knn__p": [1, 2]
}

grid_search = GridSearchCV(
    estimator=Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier())
    ]),
    param_grid=param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("\nGrid Search completed.")
print("Best parameters:")
print(grid_search.best_params_)
print("Best cross-validation accuracy:", grid_search.best_score_)

# 16. Evaluate the selected model on the test set
best_knn = grid_search.best_estimator_
y_best_pred = best_knn.predict(X_test)

final_accuracy = accuracy_score(y_test, y_best_pred)
final_precision = precision_score(
    y_test, y_best_pred, zero_division=0
)
final_recall = recall_score(
    y_test, y_best_pred, zero_division=0
)
final_f1 = f1_score(
    y_test, y_best_pred, zero_division=0
)

print("\nTuned KNN Test Metrics")
print(f"Accuracy : {final_accuracy:.4f}")
print(f"Precision: {final_precision:.4f}")
print(f"Recall   : {final_recall:.4f}")
print(f"F1 Score : {final_f1:.4f}")

classification_metrics = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
    "Score": [
        final_accuracy,
        final_precision,
        final_recall,
        final_f1
    ]
})

print("\nClassification metrics table:")
print(classification_metrics.round(4))

classification_metrics.to_csv(
    "lab5_classification_metrics.csv",
    index=False
)


# ============================================================
# PART B: KNN REGRESSION
# Predict BMI from other clinical features
# ============================================================

# 17. Prepare regression dataset
df_reg = df[df["BMI"] > 0].copy()

other_zero_cols = [
    col for col in
    ["Glucose", "BloodPressure", "SkinThickness", "Insulin"]
    if col in df_reg.columns
]

Xr = df_reg.drop(columns=["BMI", "Outcome"]).copy()
Xr = Xr.apply(pd.to_numeric, errors="coerce")
Xr[other_zero_cols] = Xr[other_zero_cols].replace(0, np.nan)

yr = pd.to_numeric(df_reg["BMI"], errors="coerce")

valid_rows = yr.notna()
Xr = Xr.loc[valid_rows]
yr = yr.loc[valid_rows]

print("\nRegression feature shape:", Xr.shape)
print("Regression target shape:", yr.shape)

# 18. Split regression data
Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    Xr,
    yr,
    test_size=0.20,
    random_state=42
)

# 19. KNN Regression pipeline
reg_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("knn", KNeighborsRegressor(n_neighbors=5))
])

reg_pipeline.fit(Xr_train, yr_train)

yr_pred = reg_pipeline.predict(Xr_test)

comparison = pd.DataFrame({
    "Actual BMI": yr_test.iloc[:10].to_numpy(),
    "Predicted BMI": np.round(yr_pred[:10], 2)
})

print("\nActual vs Predicted BMI:")
print(comparison)

# 20. Regression evaluation metrics
mae = mean_absolute_error(yr_test, yr_pred)
mse = mean_squared_error(yr_test, yr_pred)
rmse = np.sqrt(mse)
r2 = r2_score(yr_test, yr_pred)

print("\nKNN Regression Metrics")
print(f"MAE : {mae:.4f}")
print(f"MSE : {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2  : {r2:.4f}")

# 21. Effect of K on regression RMSE using cross-validation
reg_cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

rmse_cv = []

for k in range(1, 31):
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("knn", KNeighborsRegressor(n_neighbors=k))
    ])

    scores = cross_val_score(
        model,
        Xr_train,
        yr_train,
        cv=reg_cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )

    rmse_cv.append(-scores.mean())

plt.figure(figsize=(10, 6))
plt.plot(range(1, 31), rmse_cv, marker="o")
plt.xlabel("Number of Neighbors (K)")
plt.ylabel("Cross-validation RMSE")
plt.title("KNN Regression: K vs Cross-validation RMSE")
plt.xticks(range(1, 31))
plt.tight_layout()
plt.show()

best_k_reg = int(np.argmin(rmse_cv)) + 1

print("\nBest K for regression:", best_k_reg)
print("Lowest CV RMSE:", min(rmse_cv))

# 22. Tune the regression model using training data
reg_grid = GridSearchCV(
    estimator=Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("knn", KNeighborsRegressor())
    ]),
    param_grid={
        "knn__n_neighbors": list(range(3, 31, 2)),
        "knn__weights": ["uniform", "distance"],
        "knn__metric": ["euclidean", "manhattan", "minkowski"],
        "knn__p": [1, 2]
    },
    cv=reg_cv,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)

reg_grid.fit(Xr_train, yr_train)

best_reg = reg_grid.best_estimator_
yr_best_pred = best_reg.predict(Xr_test)

print("\nBest regression parameters:")
print(reg_grid.best_params_)
print("Best CV RMSE:", -reg_grid.best_score_)

print("\nTuned Regression Test Metrics")
print("MAE :", mean_absolute_error(yr_test, yr_best_pred))
print("MSE :", mean_squared_error(yr_test, yr_best_pred))
print("RMSE:", np.sqrt(mean_squared_error(yr_test, yr_best_pred)))
print("R2  :", r2_score(yr_test, yr_best_pred))

# 23. Actual vs predicted BMI plot
plt.figure(figsize=(7, 6))
plt.scatter(yr_test, yr_best_pred, alpha=0.7)

lims = [
    min(yr_test.min(), yr_best_pred.min()),
    max(yr_test.max(), yr_best_pred.max())
]

plt.plot(lims, lims, "r--", label="Perfect Prediction")
plt.xlabel("Actual BMI")
plt.ylabel("Predicted BMI")
plt.title("KNN Regression: Actual vs Predicted BMI")
plt.legend()
plt.tight_layout()
plt.show()

# 24. Save regression metrics
regression_metrics = pd.DataFrame({
    "Metric": ["MAE", "MSE", "RMSE", "R2"],
    "Score": [
        mean_absolute_error(yr_test, yr_best_pred),
        mean_squared_error(yr_test, yr_best_pred),
        np.sqrt(mean_squared_error(yr_test, yr_best_pred)),
        r2_score(yr_test, yr_best_pred)
    ]
})

print("\nRegression metrics table:")
print(regression_metrics.round(4))

regression_metrics.to_csv(
    "lab5_regression_metrics.csv",
    index=False
)

print("\nRegression metrics saved successfully.")
print("\nLab 5 execution completed successfully.")
