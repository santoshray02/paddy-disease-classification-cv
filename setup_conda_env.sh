#!/bin/bash

# Set the name of the conda environment
ENV_NAME="paddy_disease_cv"

# Create a new conda environment
conda create -n $ENV_NAME python=3.8 -y

# Activate the environment
source activate $ENV_NAME

# Install required packages
conda install -y pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install -y opencv matplotlib scikit-learn pandas numpy
conda install -y -c conda-forge jupyterlab

# Install additional packages using pip
pip install ultralytics  # for YOLOv5 and YOLOv8

echo "Conda environment '$ENV_NAME' has been created and packages have been installed."
echo "To activate the environment, use: conda activate $ENV_NAME"
