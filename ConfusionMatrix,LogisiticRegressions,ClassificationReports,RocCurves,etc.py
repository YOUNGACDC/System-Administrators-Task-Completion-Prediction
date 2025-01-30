# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 17:56:22 2024

@author: Armanis
"""

#%%
#%% Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc

# File path to the dataset
file_path = r"C:\Users\Armanis\OneDrive\Desktop\HW\SystemAdministrators.csv"

# Load the dataset
data = pd.read_csv(file_path)

# Convert 'Completed task' to a binary format (1 for Yes, 0 for No)
data['Completed task'] = data['Completed task'].map({'Yes': 1, 'No': 0})

# 1. Create a scatter plot of Experience vs. Training
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Experience', y='Training', hue='Completed task', style='Completed task', palette='deep')
plt.title('Experience vs. Training for System Administrators')
plt.xlabel('Experience (Months)')
plt.ylabel('Training Credits')
plt.legend(title='Completed Task', loc='upper right')
plt.show()
print('Based on the data it appears the more experince you have the more likely you are to complete the task')


# 2. Run a logistic regression model with both predictors using the entire dataset
# Define the features and target variable
X = data[['Experience', 'Training']]
y = data['Completed task']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)



# Get the number of false negatives
false_negatives = conf_matrix[1][0]  # True positives are in row 1, column 0
total_completed = conf_matrix[1].sum()  # Total who actually completed

# Calculate the percentage of programmers incorrectly classified
if total_completed > 0:
    percentage_incorrect = (false_negatives / total_completed) * 100
else:
    percentage_incorrect = 0

print(f"Percentage of administrators incorrectly classified as not completing the task: {percentage_incorrect:.2f}%")


#%%
# 3(optional). ROC Curve and AUC
# Get predicted probabilities
y_proba = model.predict_proba(X_test)[:, 1]  # Probabilities for class 1

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# 3. Adjust the cutoff probability
# Set your desired cutoff rate
cutoff = 0.3  # For example, a cutoff of 0.3

# Classify based on the new cutoff
y_custom_pred = (y_proba >= cutoff).astype(int)

# Evaluate accuracy with the custom cutoff
custom_accuracy = accuracy_score(y_test, y_custom_pred)
print(f'Custom Accuracy with cutoff {cutoff}: {custom_accuracy:.2f}')

# Confusion Matrix with the custom cutoff
custom_cm = confusion_matrix(y_test, y_custom_pred)
print('Confusion Matrix with custom cutoff:')
print(custom_cm)

# Classification Report with the custom cutoff
custom_report = classification_report(y_test, y_custom_pred)
print('Classification Report with custom cutoff:')
print(custom_report)

print('Decreased cutoff probabality lead to a more accurate result')

#%%


# 4. Calculate required experience for 4 years of training to exceed 0.5 probability
# Define the training value
training_value = 4  # 4 years of training (assuming the training value is in years)

# Logistic regression coefficients
coef = model.coef_[0]   # Coefficients for Experience and Training
intercept = model.intercept_[0]  # Intercept

# Calculate the experience required for probability > 0.5
# The equation derived is: 
# 0 = intercept + coef[0] * Experience + coef[1] * training_value
# Rearranging gives: Experience = -(intercept + coef[1] * training_value) / coef[0]

required_experience = -(intercept + coef[1] * training_value) / coef[0]

# Convert required experience from months to years for easier interpretation
required_experience_years = required_experience / 12  # Convert months to years

print(f"Required Experience for {training_value} years of training to exceed 0.5 probability: {required_experience:.2f} months ({required_experience_years:.2f} years)")
