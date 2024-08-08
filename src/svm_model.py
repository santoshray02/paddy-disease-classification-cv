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
    # Example usage
    X = np.random.rand(100, 10)  # 100 samples, 10 features
    y = np.random.randint(0, 3, 100)  # 3 classes
    
    svm_model = train_svm(X, y)
    
    # Make a prediction
    sample = np.random.rand(1, 10)
    prediction = predict_svm(svm_model, sample)
    print(f"Predicted class: {prediction[0]}")
