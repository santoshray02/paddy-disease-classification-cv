# Classification of Paddy Diseases Using Computer Vision Techniques

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

[Include a summary of key findings, performance metrics, and comparisons between different models]

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

Santosh Kumar

Project Link: [https://github.com/santoshray02/paddy-disease-classification-cv](https://github.com/santoshray02/paddy-disease-classification-cv)
