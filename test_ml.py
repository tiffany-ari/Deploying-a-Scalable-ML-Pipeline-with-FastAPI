import pytest
# TODO: add necessary import
import pandas as pd
import os
from ml.model import compute_model_metrics, inference, load_model, performance_on_categorical_slice, save_model, train_model
from ml.data import process_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

#create sample data and define cat_features for testing
data = pd.DataFrame({
    'age': [32, 31, 46, 56, 92],
    'workclass': ['Self-emp-not-inc', 'Private', 'Private', 'Local-gov', 'Private'],
    'education': ['HS-grad', 'Some-college', 'Bachelors', 'Masters', 'HS-grad'],
    'occupation': ['Adm-clerical', 'Prof-specialty', 'Exec-managerial', 'Craft-repair', 'Sales'],
    'race': ['White', 'Black', 'White', 'White', 'Black'],
    'sex': ['Male', 'Male', 'Female', 'Female', 'Female'],
    'hours-per-week': [40, 50, 40, 43, 58],
    'target': [0, 1, 0, 1, 0]
})

cat_features = ['workclass', 'education', 'occupation', 'race', 'sex']

train, test = train_test_split(data, test_size=0.5)
X_train, y_train, encoder, lb = process_data(train, categorical_features=cat_features, label='target', training=True)

# TODO: implement the first test. Change the function name and input as needed
def test_check_file():
    """
    tests to see if output file contains data
    """
    project_path = os.getcwd()
    data_path = os.path.join(project_path, "slice_output.txt")
    with open(data_path, "r") as file:
        file_content = file.read()

    # Check if the file contains data
    assert len(file_content) > 0, "file is empty"


# TODO: implement the second test. Change the function name and input as needed
def test_randomforest_classifier():
    """
     Test to check if trained model is type of RandomForestClassifier
    """
    # Your code here
    model = train_model(X_train, y_train)
    assert type(model).__name__ == 'RandomForestClassifier'


# TODO: implement the third test. Change the function name and input as needed
def test_randomforest_instance():
    """
     Test to check if the the train_model function returns RandomForestClassifier instance
    """
    # Your code here
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)

