"""
Instructions for running different models for the Paddy Disease Classification project.

This file contains example commands for training and running inference with various models.
Make sure you have installed all required dependencies before running these commands.

General setup:
1. Ensure you have the dataset in the 'data/paddy_doctor_dataset' directory.
2. Make sure all required packages are installed.
3. Run the commands from the project root directory.

Available models:
- ResNet50
- Inception V3
- Faster R-CNN
- RetinaNet
- SSD
- KNN
- SVM
- YOLOv5

Note: For KNN, SVM, and YOLOv5, you'll need to use their specific scripts as they have different interfaces.
"""

# ResNet50
print("Training ResNet50:")
print("python src/train.py --model_name resnet50 --data_dir data/paddy_doctor_dataset --num_epochs 10 --batch_size 32 --learning_rate 0.001")
print("\nRunning inference with ResNet50:")
print("python src/inference.py --image_path path/to/test/image.jpg --model_path resnet50_model.pth --model_name resnet50 --class_names class1 class2 class3")

# Inception V3
print("\nTraining Inception V3:")
print("python src/train.py --model_name inception_v3 --data_dir data/paddy_doctor_dataset --num_epochs 10 --batch_size 32 --learning_rate 0.001")
print("\nRunning inference with Inception V3:")
print("python src/inference.py --image_path path/to/test/image.jpg --model_path inception_v3_model.pth --model_name inception_v3 --class_names class1 class2 class3")

# Faster R-CNN
print("\nTraining Faster R-CNN:")
print("python src/train.py --model_name fasterrcnn --data_dir data/paddy_doctor_dataset --num_epochs 10 --batch_size 4 --learning_rate 0.005")
print("\nRunning inference with Faster R-CNN:")
print("python src/inference.py --image_path path/to/test/image.jpg --model_path fasterrcnn_model.pth --model_name fasterrcnn")

# RetinaNet
print("\nTraining RetinaNet:")
print("python src/train.py --model_name retinanet --data_dir data/paddy_doctor_dataset --num_epochs 10 --batch_size 4 --learning_rate 0.005")
print("\nRunning inference with RetinaNet:")
print("python src/inference.py --image_path path/to/test/image.jpg --model_path retinanet_model.pth --model_name retinanet")

# SSD
print("\nTraining SSD:")
print("python src/train.py --model_name ssd --data_dir data/paddy_doctor_dataset --num_epochs 10 --batch_size 16 --learning_rate 0.001")
print("\nRunning inference with SSD:")
print("python src/inference.py --image_path path/to/test/image.jpg --model_path ssd_model.pth --model_name ssd")

# KNN
print("\nTraining and running inference with KNN:")
print("python src/knn_model.py --data_dir data/paddy_doctor_dataset --n_neighbors 5 --test_image path/to/test/image.jpg")

# SVM
print("\nTraining and running inference with SVM:")
print("python src/svm_model.py --data_dir data/paddy_doctor_dataset --kernel rbf --test_image path/to/test/image.jpg")

# YOLOv5
print("\nTraining YOLOv5:")
print("python src/yolov5_model.py train --data data/paddy_doctor_dataset/data.yaml --img 640 --batch 16 --epochs 50 --weights yolov5s.pt")
print("\nRunning inference with YOLOv5:")
print("python src/yolov5_model.py detect --weights yolov5s.pt --img 640 --conf 0.25 --source path/to/test/image.jpg")

print("\nNote: Make sure to replace 'path/to/test/image.jpg' with the actual path to your test image.")
print("Also, ensure that you have the correct class names for classification models.")
