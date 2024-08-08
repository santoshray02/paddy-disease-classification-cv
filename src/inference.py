import torch
from torchvision import transforms
from PIL import Image
from models import get_model
from utils import load_model

def predict(image_path, model_path, class_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = get_model(len(class_names))
    load_model(model, model_path)
    model.to(device)
    model.eval()
    
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

if __name__ == "__main__":
    image_path = 'path/to/test/image.jpg'
    model_path = 'paddy_disease_classifier.pth'
    class_names = ['class1', 'class2', 'class3']  # Replace with actual class names
    
    prediction = predict(image_path, model_path, class_names)
    print(f"Predicted class: {prediction}")
