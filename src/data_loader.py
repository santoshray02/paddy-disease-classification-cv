import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CocoDetection
from pycocotools.coco import COCO

def load_data(data_dir, batch_size, model_type, train_ratio=0.8):
    """
    Load and preprocess the dataset based on the model type.
    """
    if model_type in ['resnet50', 'inception_v3', 'knn', 'svm']:
        train_loader, val_loader, test_loader, classes = load_classification_data(data_dir, batch_size, train_ratio)
        return train_loader, val_loader, classes, test_loader
    elif model_type in ['fasterrcnn', 'retinanet', 'ssd']:
        return load_object_detection_data(data_dir, batch_size, train_ratio)
    elif isinstance(model_type, str) and model_type.startswith('yolo'):
        return load_yolo_data(data_dir, batch_size, train_ratio)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def load_classification_data(data_dir, batch_size=32, train_ratio=0.7, val_ratio=0.15):
    """
    Load and preprocess data for classification models.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Assuming the dataset structure is:
    # data_dir/
    #   train/
    #     class1/
    #     class2/
    #     ...
    train_dir = os.path.join(data_dir, 'train')

    # Count the number of classes (folders) in the train directory
    num_classes = len([name for name in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, name))])
    print(f"Number of classes detected: {num_classes}")

    full_dataset = datasets.ImageFolder(root=train_dir, transform=transform)

    # Calculate sizes for train, validation, and test sets
    total_size = len(full_dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, full_dataset.classes

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

def load_yolo_data(data_dir, batch_size=32, train_ratio=0.8, yolo_version='v5'):
    """
    Prepare data for YOLO models.
    """
    # YOLO models use a different data format, typically a YAML file
    # Here we'll just return the data directory path and let the YOLO training script handle the data
    if yolo_version in ['v5', 'v6', 'v7', 'v8']:
        return data_dir
    else:
        raise ValueError(f"Unsupported YOLO version: {yolo_version}")

def collate_fn(batch):
    """
    Custom collate function for object detection data.
    """
    return tuple(zip(*batch))
