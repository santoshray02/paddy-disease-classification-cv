from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import uniform, randint
import numpy as np
import logging

def train_svm(X, y, num_epochs=10, learning_rate=0.001, kernel='rbf'):
    # Define the parameter distribution
    param_dist = {
        'C': uniform(0.1, 100),
        'max_iter': randint(num_epochs, num_epochs * 10),
        'tol': uniform(learning_rate / 10, learning_rate * 10)
    }
    
    # Create a base model
    svm = LinearSVC(dual=False)
    
    # Perform randomized search with cross-validation
    random_search = RandomizedSearchCV(svm, param_distributions=param_dist, 
                                       n_iter=20, cv=3, n_jobs=-1, verbose=1, random_state=42)
    random_search.fit(X, y)
    
    # Get the best model
    best_model = random_search.best_estimator_
    
    # Calibrate probabilities
    calibrated_svc = CalibratedClassifierCV(best_model, cv=3)
    calibrated_svc.fit(X, y)
    
    # Log the best parameters and score
    logging.info(f"Best parameters: {random_search.best_params_}")
    logging.info(f"Best cross-validation score: {random_search.best_score_:.4f}")
    
    # Evaluate the model on the entire dataset
    y_pred = calibrated_svc.predict(X)
    accuracy = accuracy_score(y, y_pred)
    
    logging.info(f"SVM Accuracy on entire dataset: {accuracy:.4f}")
    logging.info("\nClassification Report:")
    logging.info(classification_report(y, y_pred))
    
    return calibrated_svc

def predict_svm(model, X):
    return model.predict(X)

if __name__ == "__main__":
    import argparse
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    
    parser = argparse.ArgumentParser(description='Train and test SVM model.')
    args = parser.parse_args()
    
    # Load iris dataset as an example
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    svm_model = train_svm(X_train, y_train)
    
    # Evaluate on test set
    y_pred = predict_svm(svm_model, X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Make a prediction
    sample = X_test[0].reshape(1, -1)  # Use the first test sample as an example
    prediction = predict_svm(svm_model, sample)
    print(f"Predicted class: {prediction[0]}")
