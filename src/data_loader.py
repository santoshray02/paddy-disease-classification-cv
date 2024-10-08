import os
import random
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
from PIL import Image
import xml.etree.ElementTree as ET
import torchvision.transforms.functional as F

def load_data(data_dir, batch_size, model_type, train_ratio=0.8):
    """
    Load and preprocess the dataset based on the model type.
    """
    if model_type.startswith('yolo'):
        return load_yolo_data(data_dir)
    elif model_type in ['fasterrcnn', 'retinanet', 'ssd']:
        return load_object_detection_data(data_dir, batch_size, train_ratio)
    else:
        return load_classification_data(data_dir, batch_size, train_ratio)

def load_classification_data(data_dir, batch_size=32, train_ratio=0.8):
    """
    Load and preprocess data for classification models using ImageFolder format.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    except FileNotFoundError as e:
        print(f"Error loading dataset: {e}")
        print(f"Please ensure that {data_dir} exists and contains subdirectories for each class.")
        raise

    # Split the dataset
    train_size = int(train_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, None, full_dataset.classes

def load_yolo_data(data_dir):
    """
    Prepare data for YOLO models.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"The specified data directory does not exist: {data_dir}")

    # Create a YAML file for YOLO training
    yaml_content = f"""
train: {os.path.join(data_dir, 'train')}
val: {os.path.join(data_dir, 'val')}

nc: 10  # number of classes
names: ['bacterial_leaf_blight', 'bacterial_leaf_streak', 'bacterial_panicle_blight', 'blast', 'brown_spot', 'dead_heart', 'downy_mildew', 'hispa', 'normal', 'tungro']  # class names
    """
    
    yaml_path = os.path.join(data_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    return yaml_path
def load_object_detection_data(data_dir, batch_size=32, train_ratio=0.8, model_type='fasterrcnn'):
    """
    Load and preprocess data for object detection models.
    """
    def get_transform(train):
        transforms = []
        if train:
            transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.PILToTensor())
        transforms.append(T.ConvertImageDtype(torch.float))
        return transforms

    full_dataset = PaddyDiseaseDataset(data_dir, get_transform(train=True))
    
    # Split the dataset
    train_size = int(train_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_dataset.dataset.transforms = get_transform(train=True)
    val_dataset.dataset.transforms = get_transform(train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Return the number of classes without adding 1 (background is already accounted for)
    num_classes = len(full_dataset.class_to_idx)
    return train_loader, val_loader, None, num_classes

def collate_fn(batch):
    return tuple(zip(*batch))

class PaddyDiseaseDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = []
        self.labels = []
        self.class_to_idx = {}

        # Assuming the directory structure is: root/class_name/image_files
        for idx, class_name in enumerate(sorted(os.listdir(root))):
            class_dir = os.path.join(root, class_name)
            if os.path.isdir(class_dir):
                self.class_to_idx[class_name] = idx + 1  # Start labels from 1 (0 is background)
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.imgs.append(os.path.join(class_name, img_name))
                        self.labels.append(idx + 1)  # Start labels from 1 (0 is background)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.imgs[idx])
        label = self.labels[idx]
        
        img = Image.open(img_path).convert("RGB")
        
        # Create a dummy bounding box for the entire image
        w, h = img.size
        boxes = torch.tensor([[0, 0, w, h]], dtype=torch.float32)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = torch.tensor([label], dtype=torch.int64)
        
        if self.transforms is not None:
            for t in self.transforms:
                if isinstance(t, (T.PILToTensor, T.ConvertImageDtype)):
                    img = t(img)
                elif isinstance(t, T.RandomHorizontalFlip):
                    if torch.rand(1) < t.p:
                        img = F.hflip(img)
                        target["boxes"][:, [0, 2]] = w - target["boxes"][:, [2, 0]]
        
        return img, target

    def __len__(self):
        return len(self.imgs)
