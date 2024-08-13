import os
import logging
from ultralytics import YOLO
from data_loader import load_data
import torch
from models import get_model

def train(data_dir, model_name, batch_size=32, output_dir='./output', num_epochs=100, learning_rate=0.01):
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging
    log_file = os.path.join(output_dir, f'{model_name}_training.log')
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
    
    # Load data
    train_loader, val_loader, test_loader, classes = load_data(data_dir, batch_size, model_name)
    num_classes = len(classes)
    
    # Initialize model
    if model_name.startswith('yolo'):
        model = YOLO(f"{model_name}.yaml")
        results = model.train(data=train_loader, epochs=num_epochs, imgsz=640, batch=batch_size, save_dir=output_dir)
    else:
        model = get_model(model_name, num_classes=num_classes)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)
        model.to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                if model_name in ['fasterrcnn', 'retinanet', 'ssd']:
                    images, targets = batch
                    images = list(image.to(device) for image in images)
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    loss_dict = model(images, targets)
                    loss = sum(loss for loss in loss_dict.values())
                else:
                    inputs, labels = batch
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = torch.nn.functional.cross_entropy(outputs, labels)
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            logging.info(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
            
            # Evaluate on validation set
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in val_loader:
                    if model_name in ['fasterrcnn', 'retinanet', 'ssd']:
                        images, targets = batch
                        images = list(image.to(device) for image in images)
                        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                        outputs = model(images)
                        # TODO: Implement custom evaluation for object detection models
                    else:
                        inputs, labels = batch
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
            
            if model_name not in ['fasterrcnn', 'retinanet', 'ssd']:
                accuracy = 100 * correct / total
                logging.info(f"Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {accuracy:.2f}%")
        
        torch.save(model.state_dict(), os.path.join(output_dir, f'{model_name}_model.pth'))
        results = f"Training completed for {model_name}"
    
    logging.info(f"Training completed. Results: {results}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train a model on the dataset.')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--model_name', type=str, default='resnet50', help='Model to use for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save output files')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
    
    args = parser.parse_args()
    
    train(args.data_dir, args.model_name, args.batch_size, args.output_dir, args.num_epochs, args.learning_rate)
