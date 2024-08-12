import os
import sys
import logging
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data_loader import load_classification_data
from svm_model import train_svm, predict_svm
from utils import plot_training_history

def train_classifier(model, train_loader, val_loader, num_epochs, learning_rate, device, output_dir):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Clear CUDA cache before training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = amp.GradScaler()
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'logs'))
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct = 0, 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            with amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = train_correct / len(train_loader.dataset)
        
        model.eval()
        val_loss, val_correct = 0, 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                with amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = val_correct / len(val_loader.dataset)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, os.path.join(output_dir, f'best_{model.__class__.__name__}_model.pth'))
    
    writer.close()
    return history

def train_object_detection(model, train_loader, val_loader, num_epochs, learning_rate, device, output_dir):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Clear CUDA cache before training
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    scaler = amp.GradScaler()
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'logs'))
    
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
    
            optimizer.zero_grad()
            with amp.autocast():
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
    
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
    
            train_loss += losses.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for images, targets in val_loader:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        
                with amp.autocast():
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
        
                val_loss += losses.item()
        
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, os.path.join(output_dir, f'best_{model.__class__.__name__}_model.pth'))
    
    writer.close()
    return history

def train(data_dir, model_name, num_epochs=10, batch_size=32, learning_rate=0.001, output_dir='./output'):
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging
    log_file = os.path.join(output_dir, f'{model_name}_training.log')
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
    
    if model_name == 'svm':
        # Load and preprocess data
        train_data, val_data, _, classes = load_classification_data(data_dir, batch_size)
        
        # Combine train and validation data
        X = np.concatenate([batch[0].numpy() for batch in train_data] + [batch[0].numpy() for batch in val_data])
        y = np.concatenate([batch[1].numpy() for batch in train_data] + [batch[1].numpy() for batch in val_data])
        
        # Flatten the images
        X = X.reshape(X.shape[0], -1)
        
        # Scale the features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train SVM
        logging.info("Training SVM model")
        svm_model = train_svm(X_train, y_train)
        
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
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train (not used for SVM)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training (not used for SVM)')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save output files')
    
    args = parser.parse_args()
    
    train(args.data_dir, args.model_name, args.num_epochs, args.batch_size, args.learning_rate, args.output_dir)
