import pytest
import torch
import matplotlib.pyplot as plt
from src.utils import save_model, load_model, plot_training_history

@pytest.fixture
def mock_model():
    return torch.nn.Sequential(torch.nn.Linear(10, 5))

@pytest.fixture
def mock_history():
    return {
        "train_loss": [0.5, 0.4, 0.3],
        "val_loss": [0.6, 0.5, 0.4],
        "train_acc": [0.7, 0.8, 0.9],
        "val_acc": [0.6, 0.7, 0.8]
    }

def test_save_load_model(mock_model, tmp_path):
    model_path = tmp_path / "test_model.pth"
    save_model(mock_model, str(model_path))
    assert model_path.exists()

    loaded_model = torch.nn.Sequential(torch.nn.Linear(10, 5))
    load_model(loaded_model, str(model_path))
    
    # Check if the loaded model has the same parameters as the original model
    for p1, p2 in zip(mock_model.parameters(), loaded_model.parameters()):
        assert torch.all(torch.eq(p1, p2))

def test_plot_training_history(mock_history):
    fig = plot_training_history(mock_history)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)  # Close the figure to free up memory

# Add more specific tests for each function
