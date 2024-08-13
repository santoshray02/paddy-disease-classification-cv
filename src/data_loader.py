import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

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
    Load and preprocess data for classification models.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def is_valid_file(x):
        if not any(part.startswith('.') for part in os.path.normpath(x).split(os.sep)):
            return x
        return None

    try:
        full_dataset = datasets.ImageFolder(root=data_dir, transform=transform, is_valid_file=is_valid_file)
    except FileNotFoundError as e:
        print(f"Error loading dataset: {e}")
        print(f"Contents of {data_dir}:")
        for root, dirs, files in os.walk(data_dir):
            level = root.replace(data_dir, '').count(os.sep)
            indent = ' ' * 4 * level
            print(f"{indent}{os.path.basename(root)}/")
            sub_indent = ' ' * 4 * (level + 1)
            for f in files:
                print(f"{sub_indent}{f}")
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
def load_object_detection_data(data_dir, batch_size=32, train_ratio=0.8):
    """
    Load and preprocess data for object detection models.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    def is_valid_file(x):
        return not any(part.startswith('.') for part in os.path.normpath(x).split(os.sep))

    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform, is_valid_file=is_valid_file)
    
    # Split the dataset
    train_size = int(train_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    def collate_fn(batch):
        return tuple(zip(*batch))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, None, full_dataset.classes
