##########################################################################
# Importing libraries
##########################################################################
import pickle
import joblib

from environment import prepare_environment, print
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


prepare_environment()

##########################################################################
# Train a model and prepare metadata for logging
##########################################################################
# Load the Iris dataset
X, y = datasets.load_iris(return_X_y=True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define the model hyperparameters
params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "multi_class": "auto",
    "random_state": 8888,
}

# Train the model
lr = LogisticRegression(**params)
lr.fit(X_train, y_train)

# Predict on the test set
y_pred = lr.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

##########################################################################
# Serializacion using pickle
##########################################################################
model_file_name = 'store/model-v1.pkl'
# Saving the model
with open(model_file_name, 'wb') as f:
    pickle.dump(lr, f)

# Loading the serialized model from the file
with open(model_file_name, 'rb') as f:
    model = pickle.load(f)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy (After recovering using pickle): {accuracy}')


##########################################################################
# Serializacion using joblib
##########################################################################
model_file_name = 'store/model-v2.joblib'
# Save the model
joblib.dump(lr, model_file_name)  # compress=3   >> if we need small size

# Load the serialized model from HD5 file
model2 = joblib.load(model_file_name)

# Predict on the test set
y_pred = model2.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy (After recovering using joblib): {accuracy}')
