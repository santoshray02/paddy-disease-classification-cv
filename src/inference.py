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
    image_path = 'path/to/test/image.jpg'
    model_path = 'resnet50_model.pth'
    model_name = 'resnet50'
    class_names = ['class1', 'class2', 'class3']  # Replace with actual class names
    
    prediction = predict(image_path, model_path, model_name, class_names)
    
    if model_name in ['resnet50', 'inception_v3']:
        print(f"Predicted class: {prediction}")
    else:
        print(f"Prediction: {prediction}")
