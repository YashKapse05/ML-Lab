import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("titanic.csv")

# -----------------------------
# Basic Information
# -----------------------------
print(df.head())
print(df.tail())

print("Shape:", df.shape)
print(df.columns)
print(df.info())
print(df.describe())

# -----------------------------
# Missing Values
# -----------------------------
print(df.isnull().sum())

plt.figure(figsize=(8,5))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values")
plt.show()

# Fill missing values
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

print(df.isnull().sum())

# -----------------------------
# Duplicate Values
# -----------------------------
print("Duplicate Rows:", df.duplicated().sum())

# -----------------------------
# Outlier Detection
# -----------------------------
sns.boxplot(x=df["Age"])
plt.title("Age Boxplot")
plt.show()

sns.boxplot(x=df["Fare"])
plt.title("Fare Boxplot")
plt.show()

# Remove Fare Outliers
Q1 = df["Fare"].quantile(0.25)
Q3 = df["Fare"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df = df[(df["Fare"] >= lower) & (df["Fare"] <= upper)]

sns.boxplot(x=df["Fare"])
plt.title("Fare After Removing Outliers")
plt.show()

# Remove Age Outliers
Q1 = df["Age"].quantile(0.25)
Q3 = df["Age"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df = df[(df["Age"] >= lower) & (df["Age"] <= upper)]

sns.boxplot(x=df["Age"])
plt.title("Age After Removing Outliers")
plt.show()

# -----------------------------
# Encoding
# -----------------------------
df["Sex"] = df["Sex"].replace("male", 0)
df["Sex"] = df["Sex"].replace("female", 1)

print(df.head())

print(df["Embarked"].unique())

# -----------------------------
# Univariate Analysis
# -----------------------------
plt.figure(figsize=(6,4))
sns.histplot(df["Age"], bins=20, kde=True)
plt.title("Age Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df["Fare"], bins=20, kde=True)
plt.title("Fare Distribution")
plt.show()

sns.countplot(x="Sex", data=df)
plt.title("Gender Count")
plt.show()

sns.countplot(x="Pclass", data=df)
plt.title("Pclass Count")
plt.show()

sns.countplot(x="Embarked", data=df)
plt.title("Embarked Count")
plt.show()

# -----------------------------
# Bivariate Analysis
# -----------------------------
sns.countplot(x="Sex", hue="Survived", data=df)
plt.title("Gender vs Survived")
plt.show()

sns.countplot(x="Embarked", hue="Survived", data=df)
plt.title("Embarked vs Survived")
plt.show()

sns.countplot(x="Pclass", hue="Survived", data=df)
plt.title("Pclass vs Survived")
plt.show()

sns.scatterplot(x="Age", y="Fare", data=df)
plt.title("Age vs Fare")
plt.show()

# -----------------------------
# Correlation Heatmap
# -----------------------------
plt.figure(figsize=(10,8))

numeric_df = df.select_dtypes(include=["number"])

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="inferno"
)

plt.show()

# -----------------------------
# Feature Scaling
# -----------------------------
X = df.drop("Survived", axis=1)
Y = df["Survived"]

X = X.drop(["PassengerId", "Name", "Ticket"], axis=1)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X[["Age", "Fare"]] = scaler.fit_transform(X[["Age", "Fare"]])

print(X.head())

# -----------------------------
# Train Test Split
# -----------------------------
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.20,
    random_state=42
)

print("Training Data:", X_train.shape)
print("Testing Data :", X_test.shape)

print(X_train.head())
