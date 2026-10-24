# Lab 7: Naive Bayes Classification (Gaussian Naive Bayes)
# Name: Yash Kapse | Section: A | Roll No: 63 | Batch: A4

# 1. Import the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             precision_score, recall_score, f1_score)

pd.set_option('display.max_columns', None)

# 2. Load the dataset
df = pd.read_excel('Raisin_Dataset.xlsx')

# 3. Explore the dataset
df.head()
df.tail()
df.info()
df.describe()

# 4. Data quality checks and class labels
df.isnull().sum()
df.dtypes
df.duplicated().sum()
df['Class'].unique()
df['Class'].value_counts()

# 5. Bar plot of class distribution
plt.figure(figsize=(10, 5))
sns.countplot(x='Class', data=df)
plt.title('Distribution of Raisin Classes')
plt.xlabel('Raisin Class')
plt.ylabel('Number of Samples')
plt.show()

# 6. Pie chart of class distribution
class_counts = df['Class'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Raisin Classes')
plt.show()

# 7. Select the numerical features
numerical_features = df.select_dtypes(include=np.number).columns
print(numerical_features.tolist())

# 8. Histogram of numerical features
df[numerical_features].hist(figsize=(15, 10))
plt.suptitle('Distribution of Numerical Features', y=1.02, fontsize=16)
plt.tight_layout()
plt.show()

# 9. Box plot of numerical features
plt.figure(figsize=(15, 10))
for i, column in enumerate(numerical_features):
    plt.subplot(3, 3, i + 1)
    sns.boxplot(y=df[column], color='skyblue')
    plt.title(column)
plt.tight_layout()
plt.show()

# 10. Histogram with KDE for each class
for column in numerical_features:
    plt.figure(figsize=(8, 6))
    sns.histplot(data=df, x=column, hue='Class', kde=True, bins=25)
    plt.title(f'{column} Distribution by Class')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

# 11. Correlation matrix
correlation_matrix = df[numerical_features].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# 12. Scatter plots
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Area', y='Perimeter', hue='Class',
                palette={'Kecimen': 'green', 'Besni': 'blue'})
plt.title('Area vs Perimeter')
plt.xlabel('Area')
plt.ylabel('Perimeter')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='MajorAxisLength', y='MinorAxisLength', s=70, hue='Class')
plt.title('MajorAxisLength vs MinorAxisLength')
plt.show()

# 13. Data preprocessing: label encoding
label_encoder = LabelEncoder()
df['Class_encoded'] = label_encoder.fit_transform(df['Class'])
print(df[['Class', 'Class_encoded']].head(10))

print('Class Mapping:')
for index, class_name in enumerate(label_encoder.classes_):
    print(f'{index} : {class_name}')

# 14. Separate features and target
X = df[numerical_features]
y = df['Class_encoded']
print('X shape', X.shape)
print('Y shape', y.shape)

# 15. Train-test split (80:20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('Training Samples:', X_train.shape[0])
print('Testing Samples:', X_test.shape[0])

# 16. Create and train the Gaussian Naive Bayes model
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# 17. Predict on the test data
y_pred = gnb.predict(X_test)
print(y_pred)

comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison.head(10))

comparison['Actual Class'] = label_encoder.inverse_transform(comparison['Actual'])
comparison['Predicted Class'] = label_encoder.inverse_transform(comparison['Predicted'])
print(comparison.head(10))

# 18. Evaluate the model: accuracy, precision, recall, F1-score
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
print('Accuracy Percentage:', accuracy * 100, '%')

precision = precision_score(y_test, y_pred, average='weighted')
print('Precision:', precision)
print('Precision Percentage:', precision * 100, '%')

recall = recall_score(y_test, y_pred, average='weighted')
print('Recall:', recall)
print('Recall Percentage:', recall * 100, '%')

f1 = f1_score(y_test, y_pred, average='weighted')
print('F1 Score:', f1)
print('F1 Score Percentage:', f1 * 100, '%')

print('=========MODEL PERFORMANCE=========')
print('Accuracy:', accuracy * 100, '%')
print('Precision:', precision * 100, '%')
print('Recall:', recall * 100, '%')
print('F1 Score:', f1 * 100, '%')

# 19. Classification report and confusion matrix
print('Classification Report:')
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')
plt.title('Confusion Matrix - Gaussian Naive Bayes')
plt.show()

# 20. Training vs testing accuracy
y_train_pred = gnb.predict(X_train)
y_test_pred = gnb.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
train_accuracy = accuracy_score(y_train, y_train_pred)
print('Testing Accuracy:', test_accuracy)
print('Testing Accuracy Percentage:', test_accuracy * 100, '%')
print('Training Accuracy:', train_accuracy)
print('Training Accuracy Percentage:', train_accuracy * 100, '%')

# 21. Hyperparameters of GaussianNB
print(gnb.get_params())

# 22. Hyperparameter tuning for var_smoothing
smoothing_values = [1e-12, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
result = []
for var in smoothing_values:
    model = GaussianNB(var_smoothing=var)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    result.append([var,
                   accuracy_score(y_test, prediction),
                   precision_score(y_test, prediction),
                   recall_score(y_test, prediction),
                   f1_score(y_test, prediction)])

results_df = pd.DataFrame(result, columns=['var_smoothing', 'accuracy', 'precision', 'recall', 'f1'])
print(results_df)

# 23. Predict the class of a new sample (user input)
area = float(input('Enter Area:'))
major_axis = float(input('Enter MajorAxisLength:'))
minor_axis = float(input('Enter MinorAxisLength:'))
eccentricity = float(input('Enter Eccentricity:'))
convex_area = float(input('Enter ConvexArea:'))
extent = float(input('Enter Extent:'))
perimeter = float(input('Enter Perimeter:'))

new_sample = pd.DataFrame({
    'Area': [area],
    'MajorAxisLength': [major_axis],
    'MinorAxisLength': [minor_axis],
    'Eccentricity': [eccentricity],
    'ConvexArea': [convex_area],
    'Extent': [extent],
    'Perimeter': [perimeter],
})
prediction = gnb.predict(new_sample)
prediction_class = label_encoder.inverse_transform(prediction)
print('Predicted Class:', prediction_class[0])
