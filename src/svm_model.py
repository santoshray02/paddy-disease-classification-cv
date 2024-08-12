from sklearn.svm import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
import numpy as np
import logging

def train_svm_incremental(train_data, val_data, scaler, num_epochs=10, learning_rate=0.001):
    # Initialize the SGDClassifier (linear SVM)
    svm = SGDClassifier(loss='hinge', alpha=learning_rate, max_iter=1, tol=None, 
                        learning_rate='optimal', eta0=0.0, warm_start=True, random_state=42)
    
    best_accuracy = 0
    best_model = None
    
    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch + 1}/{num_epochs}")
        
        # Training
        for batch in train_data:
            X_batch, y_batch = batch
            X_batch = X_batch.numpy().reshape(X_batch.shape[0], -1)
            X_batch = scaler.partial_fit(X_batch).transform(X_batch)
            svm.partial_fit(X_batch, y_batch, classes=np.unique(y_batch))
        
        # Validation
        y_true, y_pred = [], []
        for batch in val_data:
            X_batch, y_batch = batch
            X_batch = X_batch.numpy().reshape(X_batch.shape[0], -1)
            X_batch = scaler.transform(X_batch)
            y_pred.extend(svm.predict(X_batch))
            y_true.extend(y_batch)
        
        accuracy = accuracy_score(y_true, y_pred)
        logging.info(f"Validation Accuracy: {accuracy:.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = svm
    
    # Calibrate probabilities for the best model
    calibrated_svc = CalibratedClassifierCV(best_model, cv='prefit')
    
    # Fit calibrated classifier on a subset of validation data
    X_cal, y_cal = next(iter(val_data))
    X_cal = X_cal.numpy().reshape(X_cal.shape[0], -1)
    X_cal = scaler.transform(X_cal)
    calibrated_svc.fit(X_cal, y_cal)
    
    return calibrated_svc, best_accuracy

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
