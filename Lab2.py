# ============================================================
# 1.Import Libraries & ML Libraries
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso


# ============================================================
# 2.Load Dataset
# ============================================================

df = pd.read_csv("USA_Housing.csv")


# ============================================================
# 3.Basic Information
# ============================================================

print(df.head())

print("Shape:", df.shape)

print("Columns:")
print(df.columns)

print(df.info())

print(df.describe())

print("Missing Values:")
print(df.isnull().sum())

print("Duplicate Rows:", df.duplicated().sum())

print("Data Types:")
print(df.dtypes)


# ============================================================
# 4.Exploratory Data Analysis Correlation Heatmap
# ============================================================

numeric_df = df.select_dtypes(include=['number'])

plt.figure(figsize=(10, 8))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="magma"
)

plt.title("Correlation Matrix")

plt.show()


# ============================================================
# 5. Remove Outliers Using IQR Method
# ============================================================

Q1 = df["Avg. Area Income"].quantile(0.25)
Q3 = df["Avg. Area Income"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df = df[
    (df["Avg. Area Income"] >= lower) &
    (df["Avg. Area Income"] <= upper)
]

sns.boxplot(x=df["Avg. Area Income"])
plt.show()


Q1 = df["Price"].quantile(0.25)
Q3 = df["Price"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df = df[
    (df["Price"] >= lower) &
    (df["Price"] <= upper)
]

sns.boxplot(x=df["Price"])
plt.show()


# ============================================================
# 6. Simple Linear Regression (SLR)
# ============================================================

X = df[["Avg. Area Income"]]
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

slr_model = LinearRegression()

slr_model.fit(X_train, y_train)

y_pred = slr_model.predict(X_test)

print("Simple Linear Regression")
print("------------------------")

print(
    "MAE :",
    mean_absolute_error(y_test, y_pred)
)

print(
    "MSE :",
    mean_squared_error(y_test, y_pred)
)

print(
    "RMSE :",
    np.sqrt(mean_squared_error(y_test, y_pred))
)

print(
    "R2 Score :",
    r2_score(y_test, y_pred)
)

plt.figure(figsize=(8, 5))

plt.scatter(
    X_test,
    y_test,
    label="Actual"
)

plt.plot(
    X_test,
    y_pred,
    color="red",
    label="Regression Line"
)

plt.xlabel("Avg. Area Income")
plt.ylabel("Price")
plt.title("Simple Linear Regression")

plt.legend()

plt.show()


# ============================================================
# 7 . Multiple Linear Regression (MLR)
# ============================================================

X = df[
    [
        "Avg. Area Income",
        "Avg. Area House Age",
        "Avg. Area Number of Rooms",
        "Avg. Area Number of Bedrooms",
        "Area Population"
    ]
]

y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

mlr_model = LinearRegression()

mlr_model.fit(X_train, y_train)

y_pred = mlr_model.predict(X_test)

print("Multiple Linear Regression")
print("--------------------------")

print(
    "MAE :",
    mean_absolute_error(y_test, y_pred)
)

print(
    "MSE :",
    mean_squared_error(y_test, y_pred)
)

print(
    "RMSE :",
    np.sqrt(mean_squared_error(y_test, y_pred))
)

print(
    "R2 Score :",
    r2_score(y_test, y_pred)
)

coefficients = pd.DataFrame(
    mlr_model.coef_,
    X.columns,
    columns=["Coefficient"]
)

print(coefficients)

plt.figure(figsize=(7, 6))

plt.scatter(
    y_test,
    y_pred
)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price")

plt.show()


# ============================================================
# 8 . Ridge Regression
# ============================================================

ridge_model = Ridge()

ridge_model.fit(X_train, y_train)

ridge_pred = ridge_model.predict(X_test)

print("Ridge Regression")
print("----------------")

print(
    "MAE :",
    mean_absolute_error(y_test, ridge_pred)
)

print(
    "MSE :",
    mean_squared_error(y_test, ridge_pred)
)

print(
    "RMSE :",
    np.sqrt(mean_squared_error(y_test, ridge_pred))
)

print(
    "R2 Score :",
    r2_score(y_test, ridge_pred)
)

param_grid = {
    "alpha": [0.001, 0.01, 0.1, 1, 10, 100]
}

grid_ridge = GridSearchCV(
    Ridge(),
    param_grid,
    cv=5,
    scoring="r2"
)

grid_ridge.fit(X_train, y_train)

print(
    "Best Alpha:",
    grid_ridge.best_params_
)

best_ridge = grid_ridge.best_estimator_

ridge_pred = best_ridge.predict(X_test)

print(
    "Best Ridge R2 Score:",
    r2_score(y_test, ridge_pred)
)


# ============================================================
# 9.Lasso Regression
# ============================================================

param_grid = {
    "alpha": [0.001, 0.01, 0.1, 1, 10, 100]
}

grid_lasso = GridSearchCV(
    Lasso(max_iter=5000),
    param_grid,
    cv=5,
    scoring="r2"
)

grid_lasso.fit(X_train, y_train)

print(
    "Best Alpha:",
    grid_lasso.best_params_
)

best_lasso = grid_lasso.best_estimator_

lasso_pred = best_lasso.predict(X_test)

print("Lasso Regression")
print("----------------")

print(
    "MAE :",
    mean_absolute_error(y_test, lasso_pred)
)

print(
    "MSE :",
    mean_squared_error(y_test, lasso_pred)
)

print(
    "RMSE :",
    np.sqrt(mean_squared_error(y_test, lasso_pred))
)

print(
    "R2 Score :",
    r2_score(y_test, lasso_pred)
)

print("\n========== MODEL COMPARISON ==========")

print(
    "SLR R2 Score   :",
    r2_score(
        y_test,
        slr_model.predict(
            X_test[["Avg. Area Income"]]
        )
    )
)

print(
    "MLR R2 Score   :",
    r2_score(
        y_test,
        mlr_model.predict(X_test)
    )
)

print(
    "Ridge R2 Score :",
    r2_score(
        y_test,
        ridge_pred
    )
)

print(
    "Lasso R2 Score :",
    r2_score(
        y_test,
        lasso_pred
    )
)
