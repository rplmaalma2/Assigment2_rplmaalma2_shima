import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
data = pd.read_csv('ai4i2020.csv')

# Display first few rows of the dataset
print(data.head())

data = data.drop(["UDI","Product ID","Type","TWF","HDF","PWF","OSF","RNF"], axis=1)

# Separate features and target
X = data.drop('Machine failure', axis=1)  # Features
y = data['Machine failure']  # Target variable

# Encode categorical variables if necessary
X = pd.get_dummies(X)

# Handle missing values if necessary
X = X.fillna(X.mean())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

conf_matrix = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{conf_matrix}")

class_report = classification_report(y_test, y_pred)
print(f"Classification Report:\n{class_report}")