import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CocoDetection
from pycocotools.coco import COCO
import torchvision.transforms as T
from PIL import Image

def load_data(data_dir, batch_size, model_type, train_ratio=0.8):
    """
    Load and preprocess the dataset based on the model type.
    """
    if model_type in ['resnet50', 'inception_v3', 'knn', 'svm']:
        train_loader, val_loader, test_loader, classes = load_classification_data(data_dir, batch_size, train_ratio)
        return train_loader, val_loader, test_loader, classes
    elif model_type in ['fasterrcnn', 'retinanet', 'ssd']:
        train_loader, val_loader, classes = load_object_detection_data(data_dir, batch_size, train_ratio)
        return train_loader, val_loader, None, classes
    elif isinstance(model_type, str) and model_type.startswith('yolo'):
        return load_yolo_data(data_dir, batch_size, train_ratio), None, None, None
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
    classes = [name for name in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, name))]
    num_classes = len(classes)
    print(f"Number of classes detected: {num_classes}")
    print(f"Classes: {classes}")

    full_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    
    if num_classes != len(full_dataset.classes):
        print(f"Warning: Mismatch in number of classes. Detected: {num_classes}, Dataset: {len(full_dataset.classes)}")
        print(f"Using {len(full_dataset.classes)} classes from the dataset.")
    
    num_classes = len(full_dataset.classes)

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
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Could not find data directory: {data_dir}")

    print(f"Using data directory: {data_dir}")

    # Assume the data is organized in a structure with a single 'train' directory
    train_dir = os.path.join(data_dir, 'train')
    if not os.path.exists(train_dir):
        train_dir = data_dir  # Use the main directory if 'train' doesn't exist
    
    # Get class names from subdirectories
    class_names = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    num_classes = len(class_names)
    
    # Create custom dataset
    full_dataset = CustomDataset(train_dir, class_names, transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))
    
    # Split the dataset into train and validation sets
    dataset_size = len(full_dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, class_names

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, class_names, transforms=None):
        self.root = root
        self.transforms = transforms
        self.class_names = class_names
        self.imgs = []
        self.labels = []

        for class_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(root, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.imgs.append(os.path.join(class_name, img_name))
                        self.labels.append(class_idx)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        
        if self.transforms is not None:
            img = self.transforms(img)

        return img, label

    def __len__(self):
        return len(self.imgs)

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

def load_yolo_data(data_dir):
    """
    Prepare data for YOLO models.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"The specified data directory does not exist: {data_dir}")

    # Check if the expected subdirectories exist
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')

    if not os.path.exists(train_dir):
        train_dir = data_dir  # Use the main directory if 'train' doesn't exist

    # Get the class names (assuming they are the subdirectories in the train directory)
    class_names = []
    for root, dirs, files in os.walk(train_dir):
        for dir in dirs:
            if os.path.isdir(os.path.join(root, dir)) and any(file.endswith(('.jpg', '.jpeg', '.png')) for file in os.listdir(os.path.join(root, dir))):
                class_names.append(dir)
        break  # Only process the first level of subdirectories

    if not class_names:
        raise ValueError(f"No class directories found in {train_dir}")

    # Create a YAML file for YOLO training
    yaml_content = f"""
train: {train_dir}
val: {val_dir if os.path.exists(val_dir) else train_dir}
test: {test_dir if os.path.exists(test_dir) else train_dir}

nc: {len(class_names)}  # number of classes
names: {class_names}  # class names
    """
    
    yaml_path = os.path.join(data_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    return yaml_path

def collate_fn(batch):
    """
    Custom collate function for object detection data.
    """
    return tuple(zip(*batch))
