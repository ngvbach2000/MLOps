"""
MLflow Model Training and Registration Pipeline for SVM Classification.

This module trains SVM models with different hyperparameters, logs metrics to MLflow,
and registers the best model to the Model Registry.
"""

from typing import Tuple
import mlflow
import mlflow.sklearn
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from mlflow.tracking import MlflowClient

# Constants
DEFAULT_N_SAMPLES = 1000
DEFAULT_N_FEATURES = 20
DEFAULT_N_CLASSES = 2
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_EXPERIMENT_NAME = "MLOps_Project_PhanLoai"
DEFAULT_MODEL_NAME = "PhanLoaiProject_Model"

def generate_data(
    n_samples: int = DEFAULT_N_SAMPLES,
    n_features: int = DEFAULT_N_FEATURES,
    n_classes: int = DEFAULT_N_CLASSES,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Tuple:
    """
    Generate synthetic classification dataset and split into train/test sets.
    
    Args:
        n_samples: Number of samples to generate.
        n_features: Number of features per sample.
        n_classes: Number of classes.
        test_size: Proportion of data for testing.
        random_state: Random seed for reproducibility.
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        random_state=random_state,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def train_svm_model(
    X_train, X_test, y_train, y_test, C: float = 1.0, kernel: str = "rbf"
) -> None:
    """
    Train an SVM model and log results to MLflow.
    
    Args:
        X_train: Training features.
        X_test: Test features.
        y_train: Training labels.
        y_test: Test labels.
        C: Regularization parameter.
        kernel: Kernel type (linear, rbf, poly).
    """
    with mlflow.start_run():
        # Train model
        print(f"Training SVM with C={C}, kernel={kernel}...")
        model = SVC(C=C, kernel=kernel, random_state=DEFAULT_RANDOM_STATE)
        model.fit(X_train, y_train)
        
        # Evaluate model
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        
        # Log hyperparameters
        mlflow.log_param("C", C)
        mlflow.log_param("kernel", kernel)
        mlflow.log_param("n_samples", len(X_train) + len(X_test))
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        print("Run completed and logged to MLflow.\n")

def register_best_model(experiment_name: str = DEFAULT_EXPERIMENT_NAME) -> None:
    """
    Find the best performing model and register it to MLflow Model Registry.
    
    Args:
        experiment_name: Name of the experiment to search runs from.
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        print(f"Experiment '{experiment_name}' not found. Please run training first.")
        return

    # Find all runs in the experiment, sorted by best accuracy
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.accuracy DESC"],
    )
    
    if not runs:
        print("No runs found in the experiment.")
        return

    # Get the best run
    best_run = runs[0]
    best_run_id = best_run.info.run_id
    best_accuracy = best_run.data.metrics.get("accuracy", 0.0)
    best_params = best_run.data.params
    
    print(f"Best Run ID: {best_run_id}")
    print(f"Accuracy: {best_accuracy:.4f}")
    print(f"Parameters: {best_params}")

    # Register model to Model Registry
    model_name = DEFAULT_MODEL_NAME
    model_uri = f"runs:/{best_run_id}/model"
    
    print(f"\nRegistering model '{model_name}'...")
    model_version = mlflow.register_model(model_uri, model_name)
    
    print(f"Successfully registered version {model_version.version}.")

if __name__ == "__main__":
    mlflow.set_experiment(DEFAULT_EXPERIMENT_NAME)

    # Initialize with baseline data
    X_train, X_test, y_train, y_test = generate_data()

    # Run 1: Baseline model with default parameters
    print("=" * 70)
    print("RUN 1: Baseline Model (C=1.0, kernel='rbf')")
    print("=" * 70)
    train_svm_model(X_train, X_test, y_train, y_test, C=1.0, kernel="rbf")

    # Run 2: Increase regularization parameter
    print("=" * 70)
    print("RUN 2: Tuning C parameter (C=10.0, kernel='rbf')")
    print("=" * 70)
    train_svm_model(X_train, X_test, y_train, y_test, C=10.0, kernel="rbf")

    # Run 3: Test linear kernel
    print("=" * 70)
    print("RUN 3: Linear kernel (C=1.0, kernel='linear')")
    print("=" * 70)
    train_svm_model(X_train, X_test, y_train, y_test, C=1.0, kernel="linear")

    # Run 4: Test with more data samples
    print("=" * 70)
    print("RUN 4: Increased dataset size (n_samples=2000)")
    print("=" * 70)
    X_train, X_test, y_train, y_test = generate_data(n_samples=2000)
    train_svm_model(X_train, X_test, y_train, y_test, C=1.0, kernel="rbf")

    # Register the best model
    print("=" * 70)
    print("REGISTERING BEST MODEL")
    print("=" * 70)
    register_best_model()