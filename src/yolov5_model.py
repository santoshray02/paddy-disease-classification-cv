import torch
import argparse

def load_yolov5(model_size='s'):
    model = torch.hub.load('ultralytics/yolov5', f'yolov5{model_size}')
    return model

def train_yolov5(data_yaml, model_size='s', epochs=100, batch_size=16):
    model = load_yolov5(model_size)
    model.train(data=data_yaml, epochs=epochs, batch_size=batch_size)
    return model

def predict_yolov5(model, image_path):
    results = model(image_path)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or run inference with YOLOv5.')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'predict'], help='Mode: train or predict')
    parser.add_argument('--model_size', type=str, default='s', choices=['s', 'm', 'l', 'x'], help='YOLOv5 model size')
    parser.add_argument('--data_yaml', type=str, help='Path to data.yaml file for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--image_path', type=str, help='Path to image for prediction')
    args = parser.parse_args()

    if args.mode == 'train':
        if not args.data_yaml:
            raise ValueError("data_yaml is required for training mode")
        model = train_yolov5(args.data_yaml, args.model_size, args.epochs, args.batch_size)
        model.save('yolov5_trained_model.pt')
    elif args.mode == 'predict':
        if not args.image_path:
            raise ValueError("image_path is required for prediction mode")
        model = load_yolov5(args.model_size)
        results = predict_yolov5(model, args.image_path)
        results.print()  # Print results
        results.show()  # Display results
