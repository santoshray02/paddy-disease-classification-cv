import os
import logging
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data_loader import load_classification_data
from svm_model import train_svm
from utils import plot_training_history

def train(data_dir, model_name, batch_size=32, output_dir='./output', num_epochs=10, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging
    log_file = os.path.join(output_dir, f'{model_name}_training.log')
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
    
    if model_name == 'svm':
        # Load and preprocess data
        train_data, val_data, _, classes = load_classification_data(data_dir, batch_size)
        
        # Initialize lists to store data
        X_list, y_list = [], []
        
        # Process data in batches
        for batch in train_data:
            X_batch, y_batch = batch
            X_list.append(X_batch.numpy().reshape(X_batch.shape[0], -1))
            y_list.append(y_batch.numpy())
        
        for batch in val_data:
            X_batch, y_batch = batch
            X_list.append(X_batch.numpy().reshape(X_batch.shape[0], -1))
            y_list.append(y_batch.numpy())
        
        # Combine all batches
        X = np.concatenate(X_list)
        y = np.concatenate(y_list)
        
        # Scale the features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train SVM
        logging.info("Training SVM model")
        svm_model = train_svm(X_train, y_train, num_epochs=num_epochs, learning_rate=learning_rate, kernel='rbf')
        
        # Evaluate the model
        accuracy = svm_model.score(X_test, y_test)
        logging.info(f"SVM Test Accuracy: {accuracy:.4f}")
        
        # Save the model
        import joblib
        joblib.dump(svm_model, os.path.join(output_dir, 'svm_model.joblib'))
        logging.info(f"SVM model saved to {os.path.join(output_dir, 'svm_model.joblib')}")
        
        # Plot training history (not applicable for SVM, but you can create a simple accuracy plot)
        history = {'test_accuracy': [accuracy]}
        plot_training_history(history, output_dir)
    else:
        logging.error(f"Unsupported model: {model_name}")
        raise ValueError(f"Unsupported model: {model_name}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train a model on the Paddy Doctor dataset.')
    parser.add_argument('--data_dir', type=str, default='data/paddy-disease-classification', help='Path to the dataset')
    parser.add_argument('--model_name', type=str, default='svm', choices=['svm'], help='Model to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save output files')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
    
    args = parser.parse_args()
    
    train(args.data_dir, args.model_name, args.batch_size, args.output_dir, args.num_epochs, args.learning_rate)
