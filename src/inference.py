import torch
from torchvision import transforms
from PIL import Image
from models import get_model
from utils import load_model

def predict_classifier(image_path, model, device, class_names):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image)
        _, predicted = outputs.max(1)
        
    return class_names[predicted.item()]

def predict_object_detection(image_path, model, device):
    transform = transforms.Compose([transforms.ToTensor()])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).to(device)
    
    with torch.no_grad():
        prediction = model([image])
        
    return prediction[0]

def predict(image_path, model_path, model_name, class_names=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_name in ['resnet50', 'inception_v3']:
        model = get_model(model_name, num_classes=len(class_names))
    else:
        model = get_model(model_name)
    
    load_model(model, model_path)
    model.to(device)
    model.eval()
    
    if model_name in ['resnet50', 'inception_v3']:
        return predict_classifier(image_path, model, device, class_names)
    elif model_name in ['fasterrcnn', 'retinanet', 'ssd']:
        return predict_object_detection(image_path, model, device)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Make predictions using a trained model.')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--model_name', type=str, required=True, choices=['resnet50', 'inception_v3', 'fasterrcnn', 'retinanet', 'ssd'], help='Name of the model')
    parser.add_argument('--class_names', type=str, nargs='+', help='List of class names (for classification models)')
    
    args = parser.parse_args()
    
    prediction = predict(args.image_path, args.model_path, args.model_name, args.class_names)
    
    if args.model_name in ['resnet50', 'inception_v3']:
        print(f"Predicted class: {prediction}")
    else:
        print(f"Prediction: {prediction}")
