import os
import sys
import logging

def check_imports():
    required_packages = [
        'torch', 'tqdm', 'tensorboard'
    ]
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Error: The following required packages are missing: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        sys.exit(1)

check_imports()

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from data_loader import load_data
from models import get_model
from utils import save_model, plot_training_history
import torch.cuda.amp as amp
from torch.utils.tensorboard import SummaryWriter

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
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")
    except RuntimeError as e:
        if 'CUDA' in str(e):
            logging.error(f"Error: {e}")
            logging.error("Please ensure your CUDA-enabled GPU is compatible with the current PyTorch installation.")
            logging.error("You can check the compatibility at https://pytorch.org/get-started/locally/")
            device = torch.device("cpu")
            logging.info("Falling back to CPU for training.")
        else:
            raise e
    
    if model_name in ['resnet50', 'inception_v3', 'fasterrcnn', 'retinanet', 'ssd']:
        train_loader, val_loader, test_loader, classes = load_data(data_dir, batch_size, model_name)
        num_classes = len(classes) if classes else None
        logging.info(f"Number of classes from data loader: {num_classes}")
        logging.info(f"Classes: {classes}")
        model = get_model(model_name, num_classes=num_classes)
        
        if torch.cuda.device_count() > 1:
            logging.info(f"Using {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)
        
        model = model.to(device)
        
        if model_name in ['resnet50', 'inception_v3']:
            history = train_classifier(model, train_loader, val_loader, num_epochs, learning_rate, device, output_dir)
        elif model_name in ['fasterrcnn', 'retinanet', 'ssd']:
            # Use a lower learning rate for object detection models
            od_learning_rate = learning_rate * 0.1
            logging.info(f"Adjusting learning rate for object detection: {od_learning_rate}")
            history = train_object_detection(model, train_loader, val_loader, num_epochs, od_learning_rate, device, output_dir)
        
        # You can use test_loader for final evaluation if needed
        
        save_model(model, os.path.join(output_dir, f'{model_name}_final_model.pth'))
        plot_training_history(history, output_dir)
    elif model_name in ['yolov5', 'yolov6', 'yolov7', 'yolov8']:
        yolo_module = __import__(f'{model_name}_model', fromlist=[f'train_{model_name}'])
        train_func = getattr(yolo_module, f'train_{model_name}')
        logging.info(f"Training {model_name.upper()} model")
        train_func(data_dir, epochs=num_epochs, batch_size=batch_size)
    else:
        logging.error(f"Unsupported model: {model_name}")
        raise ValueError(f"Unsupported model: {model_name}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train a model on the Paddy Doctor dataset.')
    parser.add_argument('--data_dir', type=str, default='data/paddy-disease-classification', help='Path to the dataset')
    parser.add_argument('--model_name', type=str, default='resnet50', choices=['resnet50', 'inception_v3', 'fasterrcnn', 'retinanet', 'ssd', 'yolov5', 'yolov6', 'yolov7', 'yolov8'], help='Model to train')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save output files')
    
    args = parser.parse_args()
    
    train(args.data_dir, args.model_name, args.num_epochs, args.batch_size, args.learning_rate, args.output_dir)
