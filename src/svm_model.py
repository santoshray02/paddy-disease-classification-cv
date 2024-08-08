from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

def train_svm(X, y, kernel='rbf'):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    svm = SVC(kernel=kernel)
    svm.fit(X_train, y_train)
    
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"SVM Accuracy: {accuracy:.4f}")
    
    return svm

def predict_svm(model, X):
    return model.predict(X)

if __name__ == "__main__":
    import argparse
    from sklearn.datasets import load_iris
    
    parser = argparse.ArgumentParser(description='Train and test SVM model.')
    parser.add_argument('--kernel', type=str, default='rbf', choices=['linear', 'poly', 'rbf', 'sigmoid'], help='Kernel type for SVM')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test size for train-test split')
    args = parser.parse_args()
    
    # Load iris dataset as an example
    iris = load_iris()
    X, y = iris.data, iris.target
    
    svm_model = train_svm(X, y, kernel=args.kernel)
    
    # Make a prediction
    sample = X[0].reshape(1, -1)  # Use the first sample as an example
    prediction = predict_svm(svm_model, sample)
    print(f"Predicted class: {prediction[0]}")
