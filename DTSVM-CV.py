# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold  # Added KFold here
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv(
    'dataset.csv')  # replace with your dataset path

# Assume that the last column is the target variable
X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]  # Target variable

# Data preprocessing: scaling the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Decision Tree Model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

# Support Vector Machine Model
svm_model = SVC(random_state=42)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)


# Function to print classification reports for precision, recall, f1-measure, and accuracy for each class
def print_classification_report(model_name, y_true, y_pred):
    print(f"{model_name} Classification Report:\n")
    report = classification_report(y_true, y_pred, output_dict=True)

    # Print detailed metrics for each class
    for label, metrics in report.items():
        if label not in ['accuracy', 'macro avg', 'weighted avg']:
            print(f"Class {label}:")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1-measure: {metrics['f1-score']:.4f}")
            print(f"  Accuracy: {metrics['support'] / len(y_true):.4f}")
            print("-" * 50)


# Print evaluation metrics for Decision Tree and SVM models
print_classification_report("Decision Tree", y_test, y_pred_dt)
print_classification_report("SVM", y_test, y_pred_svm)

# Evaluate models using cross-validation (on unseen data)
kf = KFold(n_splits=5, random_state=42, shuffle=True)
cv_dt = cross_val_score(dt_model, X_scaled, y, cv=kf)
cv_svm = cross_val_score(svm_model, X_scaled, y, cv=kf)

# Print cross-validation scores
print(f"Decision Tree Cross-validation Accuracy: {np.mean(cv_dt):.4f}")
print(f"SVM Cross-validation Accuracy: {np.mean(cv_svm):.4f}")

# Plot cross-validation comparison
models = ['Decision Tree', 'SVM']
cv_scores = [np.mean(cv_dt), np.mean(cv_svm)]

plt.bar(models, cv_scores, color=['blue', 'red'])
plt.xlabel('Model')
plt.ylabel('Cross-Validation Accuracy')
plt.title('Model Comparison using Cross-Validation')
plt.show()

# Investigate overfitting by comparing training and test accuracy
train_acc_dt = accuracy_score(y_train, dt_model.predict(X_train))
test_acc_dt = accuracy_score(y_test, y_pred_dt)

train_acc_svm = accuracy_score(y_train, svm_model.predict(X_train))
test_acc_svm = accuracy_score(y_test, y_pred_svm)

print(f"Decision Tree: Train Accuracy = {train_acc_dt:.4f}, Test Accuracy = {test_acc_dt:.4f}")
print(f"SVM: Train Accuracy = {train_acc_svm:.4f}, Test Accuracy = {test_acc_svm:.4f}")
