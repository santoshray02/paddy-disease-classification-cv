from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler

def train_svm_incremental(train_data, val_data, scaler, num_epochs=10, learning_rate=0.001):
    # Initialize the SGDClassifier (linear SVM)
    svm = SGDClassifier(loss='hinge', alpha=learning_rate, max_iter=1, tol=None, 
                        learning_rate='optimal', eta0=0.0, warm_start=True, random_state=42)
    
    best_accuracy = 0
    best_model = None
    
    # Get all unique classes from the entire dataset
    all_classes = np.unique(np.concatenate([y for _, y in train_data]))
    
    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch + 1}/{num_epochs}")
        
        # Training
        for batch in train_data:
            X_batch, y_batch = batch
            X_batch = X_batch.numpy().reshape(X_batch.shape[0], -1)
            X_batch = scaler.partial_fit(X_batch).transform(X_batch)
            svm.partial_fit(X_batch, y_batch, classes=all_classes)
        
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
    from torch.utils.data import TensorDataset, DataLoader
    import torch
    
    parser = argparse.ArgumentParser(description='Train and test SVM model.')
    args = parser.parse_args()
    
    # Load iris dataset as an example
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create DataLoaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize StandardScaler
    scaler = StandardScaler()
    
    # Train the model
    svm_model, accuracy = train_svm_incremental(train_loader, val_loader, scaler)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Make a prediction
    sample = X_test[0].reshape(1, -1)  # Use the first test sample as an example
    prediction = predict_svm(svm_model, scaler.transform(sample))
    print(f"Predicted class: {prediction[0]}")
