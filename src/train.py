import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from data_loader import load_data
from models import get_model
from utils import save_model, plot_training_history

def train_classifier(model, train_loader, val_loader, num_epochs, learning_rate, device):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Clear CUDA cache before training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct = 0, 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
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
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
    
    return history

def train_object_detection(model, train_loader, val_loader, num_epochs, learning_rate, device):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Clear CUDA cache before training
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            train_loss += losses.item()
        
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                val_loss += losses.item()
        
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
    
    return history

def train(data_dir, model_name, num_epochs=10, batch_size=32, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if model_name in ['resnet50', 'inception_v3', 'fasterrcnn', 'retinanet', 'ssd']:
        train_loader, val_loader, test_loader, classes = load_data(data_dir, batch_size, model_name)
        num_classes = len(classes)
        print(f"Number of classes from data loader: {num_classes}")
        print(f"Classes: {classes}")
        model = get_model(model_name, num_classes=num_classes).to(device)
        
        if model_name in ['resnet50', 'inception_v3']:
            history = train_classifier(model, train_loader, val_loader, num_epochs, learning_rate, device)
        elif model_name in ['fasterrcnn', 'retinanet', 'ssd']:
            history = train_object_detection(model, train_loader, val_loader, num_epochs, learning_rate, device)
        
        # You can use test_loader for final evaluation if needed
        
        save_model(model, f'{model_name}_model.pth')
        plot_training_history(history)
    elif model_name == 'yolov5':
        from src.yolov5_model import train_yolov5
        train_yolov5(data_dir, epochs=num_epochs, batch_size=batch_size)
    elif model_name == 'yolov6':
        from src.yolov6_model import train_yolov6
        train_yolov6(data_dir, epochs=num_epochs, batch_size=batch_size)
    elif model_name == 'yolov7':
        from src.yolov7_model import train_yolov7
        train_yolov7(data_dir, epochs=num_epochs, batch_size=batch_size)
    elif model_name == 'yolov8':
        from src.yolov8_model import train_yolov8
        train_yolov8(data_dir, epochs=num_epochs, batch_size=batch_size)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train a model on the Paddy Doctor dataset.')
    parser.add_argument('--data_dir', type=str, default='data/paddy_doctor_dataset', help='Path to the dataset')
    parser.add_argument('--model_name', type=str, default='resnet50', choices=['resnet50', 'inception_v3', 'fasterrcnn', 'retinanet', 'ssd'], help='Model to train')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
    
    args = parser.parse_args()
    
    train(args.data_dir, args.model_name, args.num_epochs, args.batch_size, args.learning_rate)
