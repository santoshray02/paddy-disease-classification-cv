import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CocoDetection
from pycocotools.coco import COCO

def load_data(data_dir, model_type, batch_size=32, train_ratio=0.8):
    """
    Load and preprocess the dataset based on the model type.
    """
    if model_type in ['resnet50', 'inception_v3', 'knn', 'svm']:
        return load_classification_data(data_dir, batch_size, train_ratio)
    elif model_type in ['fasterrcnn', 'retinanet', 'ssd']:
        return load_object_detection_data(data_dir, batch_size, train_ratio)
    elif model_type.startswith('yolo'):
        return load_yolo_data(data_dir, batch_size, train_ratio)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def load_classification_data(data_dir, batch_size=32, train_ratio=0.8):
    """
    Load and preprocess data for classification models.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    
    train_size = int(train_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, full_dataset.classes

def load_object_detection_data(data_dir, batch_size=32, train_ratio=0.8):
    """
    Load and preprocess data for object detection models.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = CocoDetection(root=f"{data_dir}/train", annFile=f"{data_dir}/train/_annotations.coco.json", transform=transform)
    val_dataset = CocoDetection(root=f"{data_dir}/valid", annFile=f"{data_dir}/valid/_annotations.coco.json", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    coco = COCO(f"{data_dir}/train/_annotations.coco.json")
    classes = [cat['name'] for cat in coco.loadCats(coco.getCatIds())]

    return train_loader, val_loader, classes

def load_yolo_data(data_dir, batch_size=32, train_ratio=0.8):
    """
    Prepare data for YOLO models.
    """
    # YOLO models use a different data format, typically a YAML file
    # Here we'll just return the data directory path and let the YOLO training script handle the data
    return data_dir

def collate_fn(batch):
    """
    Custom collate function for object detection data.
    """
    return tuple(zip(*batch))
