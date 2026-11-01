

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings("ignore")

# ## 1. Load and explore the data

df = pd.read_csv("ai.csv")
df.columns = df.columns.str.strip()

print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())
print("\nLast five rows:")
print(df.tail())
print("\nColumn names:")
print(df.columns.tolist())
print("\nData types:")
print(df.dtypes)

print(df.info())
print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())
print("\nDuplicate Rows:", df.duplicated().sum())
print("\nMachine Failure Distribution:")
print(df["Machine failure"].value_counts())

# ## 2. Features, target and train-test split

X = df.drop("Machine failure", axis=1)
y = df["Machine failure"]
print("Input shape:", X.shape)
print("Output shape:", y.shape)

categorical_features = ["Type"]
numeric_features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print("Training sample:", X_train.shape[0])
print("Testing sample:", X_test.shape[0])

# ## 3. Preprocessing pipeline

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])
print("Preprocessor created successfully!")

# ## 4. Linear SVM

linear_svm = Pipeline([
    ("preprocessor", preprocessor),
    ("svm", SVC(kernel="linear", C=1.0)),
])
linear_svm.fit(X_train, y_train)
y_pred_linear = linear_svm.predict(X_test)

accuracy_linear = accuracy_score(y_test, y_pred_linear)
precision_linear = precision_score(y_test, y_pred_linear, zero_division=0)
recall_linear = recall_score(y_test, y_pred_linear, zero_division=0)
f1_linear = f1_score(y_test, y_pred_linear, zero_division=0)

print("Linear SVM Results:")
print("Accuracy:", accuracy_linear)
print("Precision:", precision_linear)
print("Recall:", recall_linear)
print("F1-score:", f1_linear)

cm_linear = confusion_matrix(y_test, y_pred_linear)

plt.figure(figsize=(6, 4))
sns.heatmap(cm_linear, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Failure", "Failure"],
            yticklabels=["No Failure", "Failure"])
plt.title("Confusion Matrix - Linear SVM")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()

# ## 5. RBF SVM

rbf_svm = Pipeline([
    ("preprocessor", preprocessor),
    ("svm", SVC(kernel="rbf", C=1.0, gamma="scale")),
])
rbf_svm.fit(X_train, y_train)
y_pred_rbf = rbf_svm.predict(X_test)

accuracy_rbf = accuracy_score(y_test, y_pred_rbf)
precision_rbf = precision_score(y_test, y_pred_rbf, zero_division=0)
recall_rbf = recall_score(y_test, y_pred_rbf, zero_division=0)
f1_rbf = f1_score(y_test, y_pred_rbf, zero_division=0)

print("RBF SVM Results:")
print("Accuracy:", accuracy_rbf)
print("Precision:", precision_rbf)
print("Recall:", recall_rbf)
print("F1-score:", f1_rbf)

cm_rbf = confusion_matrix(y_test, y_pred_rbf)

plt.figure(figsize=(6, 4))
sns.heatmap(cm_rbf, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Failure", "Failure"],
            yticklabels=["No Failure", "Failure"])
plt.title("Confusion Matrix - RBF SVM")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()

# ## 6. Comparison of Linear and RBF kernels

comparison_plot = pd.DataFrame({
    "Kernel": ["Linear", "RBF"],
    "Accuracy": [accuracy_linear, accuracy_rbf],
    "Precision": [precision_linear, precision_rbf],
    "Recall": [recall_linear, recall_rbf],
    "F1-score": [f1_linear, f1_rbf],
})
print(comparison_plot.round(4))

comparison_plot.set_index("Kernel").plot(kind="bar", figsize=(8, 5))
plt.title("Comparison of Linear SVM and RBF SVM")
plt.xlabel("SVM Kernel")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=0)
plt.legend(title="Metrics")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# ## 7. Hyperparameter tuning with GridSearchCV

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", SVC()),
])

param_grid = [
    {
        "classifier__kernel": ["linear"],
        "classifier__C": [0.1, 1, 10, 100],
    },
    {
        "classifier__kernel": ["rbf"],
        "classifier__C": [0.1, 1, 10, 100],
        "classifier__gamma": ["scale", 0.01, 0.1, 1],
    },
]

grid_search = GridSearchCV(
    estimator=pipeline, param_grid=param_grid,
    cv=5, scoring="f1", n_jobs=-1, verbose=1
)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation F1-score:", grid_search.best_score_)

best_model = grid_search.best_estimator_
y_pred_grid = best_model.predict(X_test)

print("\nGrid Search Test Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_grid))
print("Precision:", precision_score(y_test, y_pred_grid, zero_division=0))
print("Recall:", recall_score(y_test, y_pred_grid, zero_division=0))
print("F1-score:", f1_score(y_test, y_pred_grid, zero_division=0))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_grid, zero_division=0))

cm_grid = confusion_matrix(y_test, y_pred_grid)

plt.figure(figsize=(6, 4))
sns.heatmap(cm_grid, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Failure", "Failure"],
            yticklabels=["No Failure", "Failure"])
plt.title("Confusion Matrix - Tuned SVM")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()

# ## 8. Final comparison

final_comparison = pd.DataFrame({
    "Model": ["Linear SVM", "RBF SVM", "Tuned SVM"],
    "Accuracy": [accuracy_linear, accuracy_rbf, accuracy_score(y_test, y_pred_grid)],
    "Precision": [precision_linear, precision_rbf, precision_score(y_test, y_pred_grid, zero_division=0)],
    "Recall": [recall_linear, recall_rbf, recall_score(y_test, y_pred_grid, zero_division=0)],
    "F1-score": [f1_linear, f1_rbf, f1_score(y_test, y_pred_grid, zero_division=0)],
})
final_comparison
