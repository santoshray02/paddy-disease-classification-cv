from ultralytics import YOLO
import cv2
import numpy as np

def load_yolov8(model_size='s'):
    model = YOLO(f'yolov8{model_size}.pt')
    return model

def train_yolov8(data_yaml, model_size='s', epochs=100, batch_size=16):
    model = load_yolov8(model_size)
    model.train(data=data_yaml, epochs=epochs, batch=batch_size)

def predict_yolov8(model, image):
    results = model(image)
    return results

def process_results(results, image):
    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)  # convert to RGB
        return im

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

if __name__ == "__main__":
    # Load the model
    model = load_yolov8('s')  # Load small YOLOv8 model

    # Example image path (replace with an actual image path)
    image_path = "path/to/your/image.jpg"

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image at {image_path}")
        exit(1)

    # Perform prediction
    results = predict_yolov8(model, image)

    # Process and display results
    processed_image = process_results(results, image)

    # Display the image (you might need to adjust this based on your environment)
    cv2.imshow("YOLOv8 Prediction", processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("YOLOv8 prediction completed successfully.")
