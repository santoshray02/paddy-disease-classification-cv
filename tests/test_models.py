import pytest
import torch
from src.models import PaddyDiseaseClassifier, get_model

def test_paddy_disease_classifier():
    num_classes = 10
    model = PaddyDiseaseClassifier(num_classes)
    assert isinstance(model, torch.nn.Module)
    
    # Test forward pass
    input_tensor = torch.randn(1, 3, 224, 224)
    output = model(input_tensor)
    assert output.shape == (1, num_classes)

def test_get_model():
    num_classes = 10
    model_names = ['resnet50', 'inception_v3']
    
    for model_name in model_names:
        model = get_model(model_name, num_classes)
        assert isinstance(model, torch.nn.Module)
        
        # Test forward pass
        input_tensor = torch.randn(1, 3, 224, 224)
        output = model(input_tensor)
        assert output.shape == (1, num_classes)

# Add more tests for other models as needed
