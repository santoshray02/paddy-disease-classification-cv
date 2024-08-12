from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import logging

def train_svm(X, y, kernel='rbf'):
    # Define the parameter grid
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 1],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }
    
    # Create a base model
    svm = SVC()
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(svm, param_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X, y)
    
    # Get the best model
    best_model = grid_search.best_estimator_
    
    # Log the best parameters and score
    logging.info(f"Best parameters: {grid_search.best_params_}")
    logging.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Evaluate the model on the entire dataset
    y_pred = best_model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    
    logging.info(f"SVM Accuracy on entire dataset: {accuracy:.4f}")
    logging.info("\nClassification Report:")
    logging.info(classification_report(y, y_pred))
    
    return best_model

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
