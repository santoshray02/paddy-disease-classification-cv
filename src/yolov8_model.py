from ultralytics import YOLO
import argparse

def load_yolov8(model_size='s'):
    model = YOLO(f'yolov8{model_size}.pt')
    return model

def train_yolov8(data_yaml, model_size='s', epochs=100, batch_size=16):
    model = load_yolov8(model_size)
    model.train(data=data_yaml, epochs=epochs, batch=batch_size)

def predict_yolov8(model, image_path):
    results = model(image_path)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 operations")
    parser.add_argument("--operation", choices=["train", "predict"], required=True, help="Operation to perform")
    parser.add_argument("--model_size", default="s", help="Model size (n, s, m, l, x)")
    parser.add_argument("--image_path", help="Path to image for prediction")
    parser.add_argument("--data_yaml", help="Path to data.yaml for training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for training")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    
    args = parser.parse_args()
    
    if args.operation == "train":
        train_yolov8(args.data_yaml, args.model_size, args.epochs, args.batch_size)
    elif args.operation == "predict":
        model = load_yolov8(args.model_size)
        results = predict_yolov8(model, args.image_path)
        print(results)
