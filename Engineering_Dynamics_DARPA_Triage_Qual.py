#Setup
import os
clear = lambda: os.system('cls') #on Windows System
clear()
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
import json

# Data Preprocessing
def preprocess_data(data):
    #This function preprocesses the data. The missing values are not imputed as per the requirement.

    return data

# Model Building
def build_model_GB():
    #This function builds and returns a Gradient Boosting Classifier model.
    model_gb = HistGradientBoostingClassifier()

    return model_gb

# Model Training and Predicting
def train_and_predict(model, x_train, y_train, x_test):
    #This function fits the model to the data, predicts the target for the test data, and returns predictions.
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return y_pred

# Model Evaluation
def evaluate_model(y_test, y_pred):
    #This function evaluates the model using appropriate metrics and displays results.
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model Accuracy: {accuracy}')

data_train = pd.read_csv('combined_train_data.csv')

# Preprocess data
data_train = preprocess_data(data_train)

# Build models
model_gb = build_model_GB()

# Get features and labels from data
# Assuming last column is 'target'
x_train, y_train = data_train.iloc[:,:-1], data_train.iloc[:,-1]

##Split into Test and Train Datasets
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Train and predict with Gradient Boosting
y_pred_gb = train_and_predict(model_gb, x_train, y_train, x_test)

print('Gradient Boosting Model Performance')
evaluate_model(y_test, y_pred_gb) 

data_test = pd.read_csv('combined_test_data.csv')
predictions = model_gb.predict(data_test).astype(int).tolist() 

print("Predicted Age Groups for Test Data Generated:")

# Assuming 'subject_indices' is a list of subject indices and 'predicted_classes' is a list of predicted encoded classes
subject_indices = data_test['subject'].astype(int).tolist() 

# Create a dictionary with subject indices as keys and predicted encoded classes as values
result_dict = {subject_index: encoded_class for subject_index, encoded_class in zip(subject_indices, predictions)}

# Save the dictionary to a JSON file
output_file_path = '{Engineering_Dynamics}_output.json'  # Replace with your desired file path
with open(output_file_path, 'w') as json_file:
    json.dump(result_dict, json_file, indent=4)

print(f"JSON file '{output_file_path}' has been created.")