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
