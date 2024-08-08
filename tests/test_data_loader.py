import pytest
from src.data_loader import load_data, load_classification_data, load_object_detection_data, load_yolo_data

# You might need to mock some data or use a small test dataset
@pytest.fixture
def mock_data_dir():
    return "path/to/mock/data"

def test_load_data(mock_data_dir):
    train_loader, val_loader = load_data(mock_data_dir, "classification")
    assert train_loader is not None
    assert val_loader is not None

def test_load_classification_data(mock_data_dir):
    train_loader, val_loader = load_classification_data(mock_data_dir)
    assert train_loader is not None
    assert val_loader is not None

def test_load_object_detection_data(mock_data_dir):
    train_loader, val_loader = load_object_detection_data(mock_data_dir)
    assert train_loader is not None
    assert val_loader is not None

def test_load_yolo_data(mock_data_dir):
    train_loader, val_loader = load_yolo_data(mock_data_dir)
    assert train_loader is not None
    assert val_loader is not None

# Add more specific tests for each function
