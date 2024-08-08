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
    # Download dataset and get a random image path
    image_path = download_dataset()
    print(f"Using image: {image_path}")

    # Load the model
    model = load_yolov8('s')  # Load small YOLOv8 model

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
