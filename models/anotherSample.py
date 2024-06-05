# Standard Scikit-Learn imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets

# New imports from MLFlow
import mlflow
import mlflow.sklearn


# Load the Iris dataset
X, y = datasets.load_iris(return_X_y=True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Set our tracking server uri for logging
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# Create a new MLflow Experiment
mlflow.set_experiment("MLflow RandomForestClassifier")

# Start an MLflow run assuming we have data prepared
with mlflow.start_run():
    # Build and train model
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

    # Log parameters and model implementation
    mlflow.log_param('n_estimators', rf.n_estimators)
    mlflow.sklearn.log_model(rf, 'RandomForestClassifier')

    # Evaluate model
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log the test accuracy metric
    mlflow.log_metric('accuracy', accuracy)
