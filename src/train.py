import os
import logging
import torch
from torch import nn, optim
from torchvision import models
from data_loader import load_classification_data
from utils import plot_training_history

def train(data_dir, model_name, batch_size=32, output_dir='./output', num_epochs=10, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging
    log_file = os.path.join(output_dir, f'{model_name}_training.log')
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
    
    # Load data
    train_loader, val_loader, test_loader, num_classes = load_classification_data(data_dir, batch_size)
    
    # Initialize model
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    else:
        logging.error(f"Unsupported model: {model_name}")
        raise ValueError(f"Unsupported model: {model_name}")
    
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    logging.info(f"Training {model_name} model")
    best_val_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        
        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(output_dir, f'{model_name}_best.pth'))
        
        # Log the results
        logging.info(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
    
    # Plot training history
    plot_training_history(history, output_dir)
    logging.info(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train a classification model on the Paddy Disease Classification dataset.')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--model_name', type=str, default='resnet50', help='Model to use for training (e.g., resnet50)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save output files')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
    
    args = parser.parse_args()
    
    train(args.data_dir, args.model_name, args.batch_size, args.output_dir, args.num_epochs, args.learning_rate)
