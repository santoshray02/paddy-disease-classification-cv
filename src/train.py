import os
import logging
from ultralytics import YOLO
from data_loader import load_data
import torch
from models import get_model
from torch.utils.tensorboard import SummaryWriter

def train(data_dir, model_name, batch_size=32, output_dir='./output', num_epochs=100, learning_rate=0.01):
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging
    log_file = os.path.join(output_dir, f'{model_name}_training.log')
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
    
    # Set up TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'tensorboard'))
    
    # Load data
    train_loader, val_loader, test_loader, num_classes = load_data(data_dir, batch_size, model_name)
    
    # Initialize model
    if model_name.startswith('yolo'):
        model = YOLO(f"{model_name}.yaml")
        results = model.train(data=train_loader, epochs=num_epochs, imgsz=640, batch=batch_size, save_dir=output_dir)
    else:
        model = get_model(model_name, num_classes=num_classes)
        device = torch.device('cpu')
        print(f"Using device: {device}")
        model.to(device)
        
        # For object detection models, we need to set them to training mode
        if model_name in ['fasterrcnn', 'retinanet', 'ssd']:
            model.train()
        
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)
        criterion = torch.nn.CrossEntropyLoss()
        
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for i, (images, targets) in enumerate(train_loader):
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                if model_name in ['fasterrcnn', 'retinanet', 'ssd']:
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                else:
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                    losses = loss
                
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                
                total_loss += losses.item()
                
                if i % 10 == 0:  # Log every 10 batches
                    logging.info(f"Epoch {epoch+1}/{num_epochs}, Batch {i}/{len(train_loader)}, Loss: {losses.item():.4f}")
                    writer.add_scalar('Training loss', losses.item(), epoch * len(train_loader) + i)
            
            avg_loss = total_loss / len(train_loader)
            logging.info(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
            writer.add_scalar('Average training loss', avg_loss, epoch)
            
            # Evaluate on validation set
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for images, targets in val_loader:
                    images = list(image.to(device) for image in images)
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    
                    if model_name in ['fasterrcnn', 'retinanet', 'ssd']:
                        loss_dict = model(images, targets)
                        losses = sum(loss for loss in loss_dict.values())
                    else:
                        outputs = model(images)
                        loss = criterion(outputs, targets)
                        losses = loss
                    
                    total_val_loss += losses.item()
            
            avg_val_loss = total_val_loss / len(val_loader)
            logging.info(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}")
            writer.add_scalar('Validation loss', avg_val_loss, epoch)
        
        torch.save(model.state_dict(), os.path.join(output_dir, f'{model_name}_model.pth'))
        results = f"Training completed for {model_name}"
    
    logging.info(f"Training completed. Results: {results}")
    writer.close()

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
