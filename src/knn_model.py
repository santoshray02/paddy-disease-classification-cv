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
    # Example usage
    X = np.random.rand(100, 10)  # 100 samples, 10 features
    y = np.random.randint(0, 3, 100)  # 3 classes
    
    knn_model = train_knn(X, y)
    
    # Make a prediction
    sample = np.random.rand(1, 10)
    prediction = predict_knn(knn_model, sample)
    print(f"Predicted class: {prediction[0]}")
