import os
import logging
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from data_loader import load_classification_data, load_object_detection_data
from svm_model import train_svm_incremental
from utils import plot_training_history
from ultralytics import YOLO

def train(data_dir, model_name, batch_size=32, output_dir='./output', num_epochs=10, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging
    log_file = os.path.join(output_dir, f'{model_name}_training.log')
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
    
    if model_name == 'yolov8':
        # Initialize YOLOv8 model
        model = YOLO('yolov8n.yaml')  # Create a new model from scratch
        
        # Prepare data configuration
        data_yaml = os.path.join(data_dir, 'data.yaml')
        if not os.path.exists(data_yaml):
            raise FileNotFoundError(f"data.yaml not found in {data_dir}. Please ensure your dataset is in YOLO format.")
        
        # Train YOLOv8
        logging.info("Training YOLOv8 model")
        results = model.train(
            data=data_yaml,
            epochs=num_epochs,
            imgsz=640,
            batch=batch_size,
            device=device,
            lr0=learning_rate,
            project=output_dir,
            name='yolov8_training'
        )
        
        # Save the model
        model.save(os.path.join(output_dir, 'yolov8_model.pt'))
        logging.info(f"YOLOv8 model saved to {os.path.join(output_dir, 'yolov8_model.pt')}")
        
        # Plot training history
        plot_training_history(results, output_dir)
    else:
        logging.error(f"Unsupported model: {model_name}")
        raise ValueError(f"Unsupported model: {model_name}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train a YOLOv8 model on the Paddy Doctor dataset.')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the dataset directory containing data.yaml')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save output files')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
    
    args = parser.parse_args()
    
    train(args.data_dir, 'yolov8', args.batch_size, args.output_dir, args.num_epochs, args.learning_rate)
