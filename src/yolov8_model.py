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

def load_yolov8(model_size='s'):
    model = YOLO(f'yolov8{model_size}.pt')
    return model

def train_yolov8(data_dir, model_size='s', epochs=100, batch_size=16):
    model = load_yolov8(model_size)
    
    data_yaml = 'temp_data.yaml'
    train_dir = os.path.join(data_dir, 'train_images')
    val_dir = os.path.join(data_dir, 'val_images')
    
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise ValueError(f"Training or validation directory not found in {data_dir}. Expected directories: 'train_images' and 'val_images'")
    
    data_dict = {
        'path': data_dir,
        'train': train_dir,
        'val': val_dir,
        'nc': 10,  # number of classes, adjust if needed
        'names': ['bacterial_leaf_blight', 'bacterial_leaf_streak', 'bacterial_panicle_blight', 
                  'blast', 'brown_spot', 'dead_heart', 'downy_mildew', 'hispa', 'normal', 'tungro']
    }
    
    with open(data_yaml, 'w') as f:
        yaml.dump(data_dict, f)
    
    try:
        # Train the model
        results = model.train(data=data_yaml, epochs=epochs, batch=batch_size)
        return results
    finally:
        os.remove(data_yaml)

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
    # Example usage
    data_dir = "path/to/your/data"
    
    # Train the model
    print("Training YOLOv8 model...")
    model = load_yolov8('s')
    results = train_yolov8(data_dir, model_size='s', epochs=10, batch_size=16)
    print("Training completed.")
    
    # Save the trained model
    save_model(model, "yolov8_trained.pt")
    print("Model saved.")
    
    # Load the trained model
    loaded_model = load_trained_model("yolov8_trained.pt")
    print("Trained model loaded.")
    
    # Download dataset and get a random image path for inference
    image_path = download_dataset()
    print(f"Using image for inference: {image_path}")

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image at {image_path}")
        exit(1)

    # Perform prediction
    results = predict_yolov8(loaded_model, image)

    # Process and display results
    processed_image = process_results(results, image)

    # Display the image (you might need to adjust this based on your environment)
    cv2.imshow("YOLOv8 Prediction", processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("YOLOv8 prediction completed successfully.")
    
    # Evaluate the model
    print("Evaluating the model...")
    metrics = evaluate_yolov8(loaded_model, "temp_data.yaml")
    print("Evaluation metrics:", metrics)
