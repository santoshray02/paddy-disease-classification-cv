import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.detection import fasterrcnn_resnet50_fpn, retinanet_resnet50_fpn
from torchvision.models.detection.ssd import ssd300_vgg16

class PaddyDiseaseClassifier(nn.Module):
    def __init__(self, num_classes, model_name='resnet50'):
        super(PaddyDiseaseClassifier, self).__init__()
        if model_name == 'resnet50':
            self.base_model = models.resnet50(pretrained=True)
            num_ftrs = self.base_model.fc.in_features
            self.base_model.fc = nn.Linear(num_ftrs, num_classes)
        elif model_name == 'inception_v3':
            self.base_model = models.inception_v3(pretrained=True)
            num_ftrs = self.base_model.fc.in_features
            self.base_model.fc = nn.Linear(num_ftrs, num_classes)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        print(f"Initialized {model_name} with {num_classes} output classes")
        self.num_classes = num_classes  # Store the number of classes

    def forward(self, x):
        return self.base_model(x)

def get_model(model_name, num_classes=None, pretrained=True):
    if model_name == 'resnet50' or model_name == 'inception_v3':
        return PaddyDiseaseClassifier(num_classes, model_name)
    elif model_name == 'fasterrcnn':
        return fasterrcnn_resnet50_fpn(pretrained=pretrained)
    elif model_name == 'retinanet':
        return retinanet_resnet50_fpn(pretrained=pretrained)
    elif model_name == 'ssd':
        return ssd300_vgg16(pretrained=pretrained)
    elif model_name == 'yolov5':
        from src.yolov5_model import load_yolov5
        return load_yolov5()
    elif model_name == 'yolov6':
        from src.yolov6_model import load_yolov6
        return load_yolov6()
    elif model_name == 'yolov7':
        from src.yolov7_model import load_yolov7
        return load_yolov7()
    elif model_name == 'yolov8':
        from src.yolov8_model import load_yolov8
        return load_yolov8()
    else:
        raise ValueError(f"Unsupported model: {model_name}")

# Note: KNN and SVM are not deep learning models and will be implemented separately.
# YOLO models (v5, v6, v7, v8) require external libraries and will be implemented in separate files.
