# Anomaly Detection Algorithm for the Search for New Physics

**Author:** Andrés Felipe Duque Bran

***

## Overview

This repository contains a collection of scripts and modules for preprocessing, training, and evaluating an anomaly detection model applied on particle physics, specifically on the [LHC Olympics 2020 challenge](https://lhco2020.github.io/homepage/). The main components include data preprocessing, autoencoder training, and anomaly detection using BumpHunter. The external dataset is prepared via a Bash script, and various Python scripts are utilized for specific analysis tasks.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Model Training and Testing](#model-training-and-testing)
  - [BumpHunter Analysis](#bumphunter-analysis)
- [Repository Structure](#repository-structure)

## Installation

### Prerequisites

- Python 3.9 or higher
- Git

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/afduquebr/Autoencoder.git
   cd Autoencoder
   ```

2. **Prepare External Dataset**:

   **Note**: The `setup_dataset.sh` script should only be run **once** immediately after cloning the repository. This script clones, preprocesses and sets up the necessary datasets from the LHC Olympics challenge.

   ```bash
   chmod +x setup_dataset.sh
   ./setup_dataset.sh
   ```

   This script will clone the [dataset repository](https://gitlab.cern.ch/idinu/clustering-lhco) in the directory `../LHCO-Dataset`, set up a virtual environment, modify necessary files due to dependencies deprecation, and perform data clustering.

3. **Install Python dependencies**:

   Activate the virtual environment and install dependencies:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

## Usage

### Data Preparation

1. **Preprocessing**: The `Preprocessor` class defined in `preprocessing.py` standardizes and prepare data for training. This class is already included in the scripts for training testing and applying the algorithm.

### Model Training and Testing

1. **Training**: Define and train the autoencoder model using the `AutoEncoder` class. Use `train.py` to handle the training pipeline. The command for training the model with 0.5\% of RnD signal is shown below:

   ```bash
   python3 train.py --dataset sig --anomaly 0.5
   ```

2. **Testing and Evaluation**: The `test.py` script evaluates the model's performance on the test set. It includes calculating reconstruction errors, plotting histograms, and computing metrics like ROC curves and signal efficiencies. The command for testing the model with 0.5\% of RnD signal is shown below:

   ```bash
   python3 test.py --dataset sig --anomaly 0.5
   ```

3. **Histogram Plotting**: The `hist.py` script handles the generation of histograms of the dataset and signal before and after passing through the model. The command for obtaining the histograms with 0.5\% of RnD signal is shown below:

   ```bash
   python3 hist.py --dataset sig --anomaly 0.5
   ```


### BumpHunter Analysis

1. **Anomaly Detection**: The `apply.py` script implements the BumpHunter algorithm to identify significant deviations in the mass spectrum, indicating potential new physics signals. The command for applying the BumpHunter algorithm on the model with 0.5\% of RnD signal is shown below:

   ```bash
   python3 apply.py --dataset sig --anomaly 0.5
   ```

## Repository Structure

```
Autoencoder
│
├── autoencoder.py           # Autoencoder model and loss functions
├── preprocessing.py         # Data preprocessing utilities
├── main.py                  # Script for specifying signal insertion
├── train.py                 # Main script for training the autoencoder
├── test.py                  # Script for testing and generating results
├── apply.py                 # Script for the BumpHunter analysis
├── setup_dataset.sh         # Bash script to prepare LHCO datasets
├── README.md                # This README file
├── requirements.txt         # Python dependencies
├── figs/                    # Training, testing and BH analysis figures
├── models/                  # Directory containing trained models
└── selection/               # Directory containing feature selection files
```


***

*This work was developed as an internship project for the completion of the programme of Master 2 Fundamental Physics and Applications in the path of Universe and Particles at the* **_Université Clermont Auvergne_***, in collaboration with the* **_Laboratoire de Physique de Clermont Auvergne_***. Its development was performed under the supervision of professors Julien Donini and Samuel Calvet.*