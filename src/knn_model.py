from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

def train_knn(X, y, n_neighbors=5):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"KNN Accuracy: {accuracy:.4f}")
    
    return knn

def predict_knn(model, X):
    return model.predict(X)

if __name__ == "__main__":
    import argparse
    from sklearn.datasets import load_iris
    
    parser = argparse.ArgumentParser(description='Train and test KNN model.')
    parser.add_argument('--n_neighbors', type=int, default=5, help='Number of neighbors for KNN')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test size for train-test split')
    args = parser.parse_args()
    
    # Load iris dataset as an example
    iris = load_iris()
    X, y = iris.data, iris.target
    
    knn_model = train_knn(X, y, n_neighbors=args.n_neighbors)
    
    # Make a prediction
    sample = X[0].reshape(1, -1)  # Use the first sample as an example
    prediction = predict_knn(knn_model, sample)
    print(f"Predicted class: {prediction[0]}")
