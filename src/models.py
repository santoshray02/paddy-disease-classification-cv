import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.detection import fasterrcnn_resnet50_fpn, retinanet_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.ssd import ssd300_vgg16

class PaddyDiseaseClassifier(nn.Module):
    def __init__(self, num_classes, model_name='resnet50'):
        super(PaddyDiseaseClassifier, self).__init__()
        self.device = torch.device("cpu")
        if model_name == 'resnet50':
            self.base_model = models.resnet50(pretrained=True)
            num_ftrs = self.base_model.fc.in_features
            self.base_model.fc = nn.Linear(num_ftrs, num_classes)
        elif model_name == 'inception_v3':
            self.base_model = models.inception_v3(pretrained=True, aux_logits=True)
            self.base_model.aux_logits = False
            self.base_model.AuxLogits = None
            self.base_model.Conv2d_1a_3x3.conv = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
            self.base_model.Conv2d_2a_3x3.conv = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
            self.base_model.Conv2d_2b_3x3.conv = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
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
        model = fasterrcnn_resnet50_fpn(pretrained=pretrained)
        if num_classes is not None:
            # Get number of input features for the classifier
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            # Replace the pre-trained head with a new one
            # num_classes already includes the background class
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model
    elif model_name == 'retinanet':
        model = retinanet_resnet50_fpn(pretrained=pretrained)
        if num_classes is not None:
            # Get number of input features for the classifier
            in_features = model.head.classification_head.cls_logits.in_channels
            num_anchors = model.head.classification_head.num_anchors
            # Replace the pre-trained head with a new one
            model.head.classification_head.num_classes = num_classes
            model.head.classification_head.cls_logits = nn.Conv2d(in_features, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
        return model
    elif model_name == 'ssd':
        model = ssd300_vgg16(pretrained=pretrained)
        if num_classes is not None:
            # Replace classification head
            num_anchors = model.anchor_generator.num_anchors_per_location()[0]
            model.head.classification_head = nn.Conv2d(model.backbone.out_channels, num_anchors * num_classes, kernel_size=3, padding=1)
        return model
    else:
        raise ValueError(f"Unsupported model: {model_name}")

# Note: KNN and SVM are not deep learning models and will be implemented separately.
# YOLO models (v5, v6, v7, v8) require external libraries and will be implemented in separate files.
