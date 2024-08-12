import os
import logging
from ultralytics import YOLO
from data_loader import load_yolo_data, load_object_detection_data, load_classification_data, collate_fn
import torchvision
import torch
from torchvision.models.detection import retinanet_resnet50_fpn, RetinaNet
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.retinanet import RetinaNet_ResNet50_FPN_Weights

def train(data_dir, model_name, batch_size=32, output_dir='./output', num_epochs=100, learning_rate=0.01, num_classes=None):
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging
    log_file = os.path.join(output_dir, f'{model_name}_training.log')
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
    
    # Load data
    try:
        if model_name.startswith('yolo'):
            data_yaml = load_yolo_data(data_dir)
        elif model_name in ['resnet50', 'inception_v3']:
            train_loader, val_loader, test_loader, classes = load_classification_data(data_dir, batch_size)
            num_classes = len(classes)
        else:
            train_loader, val_loader, classes = load_object_detection_data(data_dir, batch_size)
            num_classes = len(classes)
    except FileNotFoundError as e:
        logging.error(f"Error: The specified data directory does not exist: {data_dir}")
        raise
    except ValueError as e:
        logging.error(f"Error: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error loading data: {str(e)}")
        raise

    if num_classes is None:
        num_classes = len(classes)
        logging.info(f"Number of classes determined from the dataset: {num_classes}")
    
    # Initialize model
    if model_name.startswith('yolov8'):
        model = YOLO(f"{model_name}.yaml")
    elif model_name.startswith('yolov5'):
        model = YOLO(f"{model_name}.pt")
    elif model_name.startswith('yolov6'):
        from yolov6 import YOLOv6
        model = YOLOv6(model_name)
    elif model_name.startswith('yolov7'):
        from yolov7 import YOLOv7
        model = YOLOv7(model_name)
    elif model_name == 'retinanet':
        train_loader, val_loader, classes = load_object_detection_data(data_dir, batch_size)
        num_classes = len(classes)
        
        # Load a pre-trained model
        model = retinanet_resnet50_fpn(weights=RetinaNet_ResNet50_FPN_Weights.DEFAULT)
        
        # Modify the classification head for the new number of classes
        in_features = model.backbone.out_channels
        num_anchors = model.anchor_generator.num_anchors_per_location()[0]
        model.head.classification_head.num_classes = num_classes
        model.head.classification_head.cls_logits = torch.nn.Conv2d(in_features, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
        
        # Modify the box regression head
        model.head.regression_head.bbox_reg = torch.nn.Conv2d(in_features, num_anchors * 4, kernel_size=3, stride=1, padding=1)
    else:
        logging.error(f"Unsupported model: {model_name}")
        raise ValueError(f"Unsupported model: {model_name}")
    
    # Train the model
    logging.info(f"Training {model_name} model")
    
    if model_name.startswith(('yolov8', 'yolov5')):
        results = model.train(
            data=data_yaml,
            epochs=num_epochs,
            imgsz=640,
            batch=batch_size,
            save_dir=output_dir,
            lr0=learning_rate
        )
    elif model_name.startswith('yolov6'):
        results = model.train(
            data_yaml=data_yaml,
            epochs=num_epochs,
            batch_size=batch_size,
            img_size=640,
            output_dir=output_dir,
            lr=learning_rate
        )
    elif model_name.startswith('yolov7'):
        results = model.train(
            data=data_yaml,
            epochs=num_epochs,
            batch_size=batch_size,
            img_size=640,
            project=output_dir,
            hyp={'lr0': learning_rate}
        )
    elif model_name == 'retinanet':
        train_loader, val_loader, _ = load_object_detection_data(data_dir, batch_size)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for images, targets in train_loader:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} if isinstance(t, dict) else t.to(device) for t in targets]
        
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                total_loss += losses.item()
        
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
        
            avg_loss = total_loss / len(train_loader)
            logging.info(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
            
            # Evaluate on validation set
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for images, targets in val_loader:
                    images = list(image.to(device) for image in images)
                    targets = [{k: v.to(device) for k, v in t.items()} if isinstance(t, dict) else t.to(device) for t in targets]
                
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    val_loss += losses.item()
            
            avg_val_loss = val_loss / len(val_loader)
            logging.info(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}")
            
            lr_scheduler.step()
        
        torch.save(model.state_dict(), os.path.join(output_dir, 'retinanet_model.pth'))
        results = "Training completed for RetinaNet"
    
    logging.info(f"Training completed. Results: {results}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train a YOLO model on the dataset.')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--model_name', type=str, default='yolov8n', help='Model to use for training (e.g., yolov8n, yolov5s, yolov6n, yolov7, retinanet)')
    parser.add_argument('--num_classes', type=int, help='Number of classes (required for RetinaNet)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save output files')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for training')
    
    args = parser.parse_args()
    
    train(args.data_dir, args.model_name, args.batch_size, args.output_dir, args.num_epochs, args.learning_rate, args.num_classes)
