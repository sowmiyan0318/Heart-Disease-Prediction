# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
# Load the dataset (replace 'heart_disease.csv' with your dataset)
# Example dataset can be found here: https://www.kaggle.com/ronitf/heart-disease-uci
df = pd.read_csv('heart_disease.csv.csv')

# Preview the dataset
print(df.head())
# Identify categorical columns (e.g., 'sex', 'cp', etc.)
categorical_columns = ['sex', 'dataset', 'cp', 'exang', 'slope', 'thal','ca', 'restecg']  # Example list, modify as needed

# Convert categorical columns to numerical using one-hot encoding
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
# Check for missing values
print(df.isnull().sum())

# Features and target variable
X = df.drop('num', axis=1)  # Features (excluding the target column)
y = df['num']  # Target variable (0 = No disease, 1 = Disease)


# Convert to DataFrame (optional, for easier manipulation)


# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (important for algorithms like Logistic Regression)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Initialize the Logistic Regression model
model = LogisticRegression()

# Train the model on the training data
model.fit(X_train_imputed, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test_imputed)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Print classification report
print(classification_report(y_test, y_pred))

# Print confusion matrix
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

# Example of predicting a new patient's likelihood of heart disease
# (Use real data in place of this random example)
new_patient = np.array([[25, 120, 80, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0.5] + [0]*11])  # Replace with the actual features of the new patient
new_patient_scaled = scaler.transform(new_patient)
prediction = model.predict(new_patient_scaled)

if prediction == 1:
    print('The model predicts the patient is at risk of heart disease.')
else:
    print('The model predicts the patient is not at risk of heart disease.')
