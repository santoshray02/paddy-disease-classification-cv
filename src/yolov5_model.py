import torch

def load_yolov5(model_size='s'):
    model = torch.hub.load('ultralytics/yolov5', f'yolov5{model_size}')
    return model

def train_yolov5(data_yaml, model_size='s', epochs=100, batch_size=16):
    model = load_yolov5(model_size)
    model.train(data=data_yaml, epochs=epochs, batch_size=batch_size)

def predict_yolov5(model, image_path):
    results = model(image_path)
    return results

if __name__ == "__main__":
    # Example usage
    model = load_yolov5()
    
    # For training (uncomment and modify as needed)
    # train_yolov5('path/to/data.yaml')
    
    # For inference
    results = predict_yolov5(model, 'path/to/image.jpg')
    results.print()  # Print results
    results.show()  # Display results
