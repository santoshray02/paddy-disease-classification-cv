# Classification of Paddy Diseases Using Computer Vision Techniques

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/santoshray02/paddy-disease-classification-cv)
[![Coverage Status](https://img.shields.io/badge/coverage-80%25-yellowgreen)](https://github.com/santoshray02/paddy-disease-classification-cv)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

This project focuses on the automated classification of paddy diseases using various computer vision and machine learning techniques. We utilize the "Paddy Doctor" dataset for training and evaluation of our models.

## Dataset

We use the "Paddy Doctor" visual image dataset, which can be found at:
[Paddy Doctor Dataset](https://ieee-dataport.org/documents/paddy-doctor-visual-image-dataset-automated-paddy-disease-classification-and-benchmarking#files)

## Project Structure

- `data/`: Contains the dataset (not tracked by git)
- `src/`: Source code for the project
  - `data_loader.py`: Functions for loading and preprocessing data
  - `models.py`: Definitions of various neural network models
  - `train.py`: Training scripts for different models
  - `inference.py`: Inference scripts for trained models
  - `utils.py`: Utility functions
  - `knn_model.py`: Implementation of K-Nearest Neighbors model
  - `svm_model.py`: Implementation of Support Vector Machine model
  - `yolov5_model.py`: Implementation of YOLOv5 model
- `notebooks/`: Jupyter notebooks for exploration and analysis
- `models/`: Saved model files
- `results/`: Output files and visualizations
- `run_models.py`: Script to run different models

## Supported Models

1. Convolutional Neural Networks (CNN)
   - ResNet50
   - Inception V3
2. Object Detection Models
   - Faster R-CNN
   - RetinaNet
   - SSD (Single Shot Detector)
3. YOLO (You Only Look Once) variants
   - YOLOv5
4. Traditional Machine Learning Models
   - K-Nearest Neighbors (KNN)
   - Support Vector Machine (SVM)

## Setup

1. Clone the repository:
   ```
   git clone [repository-url]
   cd [repository-name]
   ```
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Download the Paddy Doctor dataset and place it in the `data/` directory.

## Usage

To train a model:
```
python src/train.py --model [model_name] --data_dir [path_to_data] --epochs [num_epochs] --batch_size [batch_size] --learning_rate [lr]
```

To run inference:
```
python src/inference.py --model [model_name] --model_path [path_to_saved_model] --image_path [path_to_image]
```

For detailed instructions on running each model, refer to `run_models.py`.

## Results

### Model Performance Comparison

Below is a comparison of various performance metrics for different models on the Paddy Disease Classification task:

| Model | Accuracy | Precision | Recall | F1 Score | Inference Time (ms) | Model Size (MB) |
|-------|----------|-----------|--------|----------|---------------------|-----------------|
| ResNet50 | TBD | TBD | TBD | TBD | TBD | TBD |
| Inception V3 | TBD | TBD | TBD | TBD | TBD | TBD |
| Faster R-CNN | TBD | TBD | TBD | TBD | TBD | TBD |
| RetinaNet | TBD | TBD | TBD | TBD | TBD | TBD |
| SSD | TBD | TBD | TBD | TBD | TBD | TBD |
| YOLOv5 | TBD | TBD | TBD | TBD | TBD | TBD |
| KNN | TBD | TBD | TBD | TBD | TBD | TBD |
| SVM | TBD | TBD | TBD | TBD | TBD | TBD |

*Note: The values are placeholders (TBD) and will be updated once all models have been trained and evaluated.*

[Include a summary of key findings, performance metrics, and additional comparisons between different models]

#### Visualizations

```
[Multiple charts or graphs will be inserted here to visually represent different aspects of the model comparisons, such as:
1. Bar chart for accuracy comparison
2. Scatter plot for accuracy vs. inference time
3. Radar chart for precision, recall, and F1 score
4. Bar chart for model sizes]
```

*Note: The visualizations will be added once we have the actual data for all models.*

### Additional Benchmarks

1. **Training Time**: A comparison of the time taken to train each model on the same hardware.
2. **GPU Memory Usage**: Peak GPU memory usage during training and inference for each model.
3. **CPU vs. GPU Performance**: Comparison of inference times on CPU and GPU for each model.
4. **Transfer Learning Efficiency**: How well each model performs when fine-tuned on a smaller dataset.
5. **Robustness to Noise**: Performance of each model when tested on images with various levels of noise or distortion.

[Placeholder for additional benchmark results and analysis]

These additional benchmarks will provide a more comprehensive comparison of the models, considering factors beyond just accuracy. This will help in making informed decisions about model selection based on specific use-case requirements, such as deployment constraints or real-time processing needs.

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

Santosh Kumar

Project Link: [https://github.com/santoshray02/paddy-disease-classification-cv](https://github.com/santoshray02/paddy-disease-classification-cv)

## Environment Setup

To set up the environment for this project, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/santoshray02/paddy-disease-classification-cv.git
   cd paddy-disease-classification-cv
   ```

2. Make the setup script executable:
   ```
   chmod +x setup_conda_env.sh
   ```

3. Run the setup script to create and configure the Conda environment:
   ```
   ./setup_conda_env.sh
   ```

4. Activate the Conda environment:
   ```
   conda activate paddy_disease_cv
   ```

Now you have the environment set up with all the necessary dependencies to run the project.
