from ultralytics import YOLO
import cv2
import numpy as np
import os
import random
import requests
from zipfile import ZipFile
from io import BytesIO
import yaml
import torch
import argparse

def load_yolov8(model_size='s'):
    model = YOLO(f'yolov8{model_size}.pt')
    return model

def train_yolov8(data_yaml, model_size='s', epochs=100, batch_size=16, learning_rate=0.01):
    model = load_yolov8(model_size)
    
    try:
        # Ensure data_yaml path is absolute
        data_yaml = os.path.abspath(data_yaml)
        
        # Check if the YAML file exists
        if not os.path.exists(data_yaml):
            raise FileNotFoundError(f"The data YAML file '{data_yaml}' does not exist.")
        
        # Load and validate the YAML file
        with open(data_yaml, 'r') as file:
            yaml_data = yaml.safe_load(file)
        
        # Ensure train and val paths are absolute
        for key in ['train', 'val']:
            if key in yaml_data:
                yaml_data[key] = os.path.abspath(os.path.join(os.path.dirname(data_yaml), yaml_data[key]))
                if not os.path.exists(yaml_data[key]):
                    raise FileNotFoundError(f"The {key} path '{yaml_data[key]}' does not exist.")
        
        # Write the updated YAML data to a temporary file
        temp_yaml = 'temp_data.yaml'
        with open(temp_yaml, 'w') as file:
            yaml.dump(yaml_data, file)
        
        # Train the model using the temporary YAML file
        results = model.train(data=temp_yaml, epochs=epochs, batch=batch_size, lr0=learning_rate)
        
        # Remove the temporary YAML file
        os.remove(temp_yaml)
        
        return results, model
    except Exception as e:
        print(f"An error occurred during training: {e}")
        return None, None

def predict_yolov8(model, image):
    results = model(image)
    return results

def process_results(results, image):
    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)  # convert to RGB
        return im

def save_model(model, path):
    model.save(path)

def load_trained_model(path):
    return YOLO(path)

def evaluate_yolov8(model, data_yaml):
    """
    Evaluate YOLOv8 model and return metrics.
    
    Args:
    model: YOLO model object
    data_yaml (str): Path to the data.yaml file
    
    Returns:
    dict: Dictionary containing metrics including mAP and F1 score
    """
    results = model.val(data=data_yaml)
    metrics = results.results_dict
    return metrics

def download_dataset():
    url = "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip"
    r = requests.get(url)
    with ZipFile(BytesIO(r.content)) as zip_ref:
        zip_ref.extractall("dataset")
    
    image_dir = "dataset/coco128/images/train2017"
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    random_image = random.choice(image_files)
    return os.path.join(image_dir, random_image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 operations")
    parser.add_argument("--operation", type=str, required=True, choices=['train', 'predict', 'evaluate'], help="Operation to perform")
    parser.add_argument("--data_yaml", type=str, help="Path to data.yaml file")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for training")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for training")
    parser.add_argument("--model_size", type=str, default='s', choices=['n', 's', 'm', 'l', 'x'], help="YOLOv8 model size")
    parser.add_argument("--image_path", type=str, help="Path to image for prediction")
    parser.add_argument("--model_path", type=str, default="yolov8_trained.pt", help="Path to save/load the model")

    args = parser.parse_args()

    if args.operation == 'train':
        print("Training YOLOv8 model...")
        results, model = train_yolov8(args.data_yaml, model_size=args.model_size, epochs=args.epochs, 
                                      batch_size=args.batch_size, learning_rate=args.learning_rate)
        if results is not None:
            print("Training completed.")
            save_model(model, args.model_path)
            print(f"Model saved to {args.model_path}")
        else:
            print("Training failed.")

    elif args.operation == 'predict':
        if not args.image_path:
            print("Error: --image_path is required for prediction")
            exit(1)

        model = load_trained_model(args.model_path)
        print(f"Model loaded from {args.model_path}")

        image = cv2.imread(args.image_path)
        if image is None:
            print(f"Error: Unable to read image at {args.image_path}")
            exit(1)

        results = predict_yolov8(model, image)
        processed_image = process_results(results, image)

        cv2.imshow("YOLOv8 Prediction", processed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print("YOLOv8 prediction completed successfully.")

    elif args.operation == 'evaluate':
        if not args.data_yaml:
            print("Error: --data_yaml is required for evaluation")
            exit(1)

        model = load_trained_model(args.model_path)
        print(f"Model loaded from {args.model_path}")

        print("Evaluating the model...")
        metrics = evaluate_yolov8(model, args.data_yaml)
        print("Evaluation metrics:", metrics)

    else:
        print(f"Unknown operation: {args.operation}")
