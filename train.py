#######################################################################################################
"""
Created on Jun 10 2024

@author: Andres Felipe DUQUE BRAN
"""
#######################################################################################################

import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from autoencoder import AutoEncoder, WeightedMSELoss, train, test
from preprocessing import Preprocessor
from main import parse_args

####################################### GPU or CPU running ###########################################

# Determine the device for computations (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#######################################################################################################
####################################### Data Preprocessing ###########################################

# Parse command line arguments for signal and percentage
signal, pct = parse_args()
# Initialize the Preprocessor and retrieve the scaled data and weights
preprocessing = Preprocessor()
sample_scaled, _, sig_scaled = preprocessing.get_scaled_data()
weights_sample = preprocessing.get_weights()
mjj_sample, _, _ = preprocessing.get_mass()

#######################################################################################################
######################################## Data in Tensors ###########################################

# Convert data to torch tensors and move to the selected device
train_sample = torch.from_numpy(sample_scaled[:100000].values).float().to(device)
test_sample = torch.from_numpy(sample_scaled[100000:200000].values).float().to(device)
test_sig = torch.from_numpy(sig_scaled[:100000].values).float().to(device)
weights = torch.from_numpy(weights_sample[:100000]).float().to(device)
mjj = torch.from_numpy(mjj_sample[:100000]).float().to(device)

# Create TensorDatasets for training, testing, and signal data
trainSet = TensorDataset(train_sample, weights, mjj)
testSet = TensorDataset(test_sample, test_sample)
testSet_sig = TensorDataset(test_sig, test_sig)

#######################################################################################################
######################################### Model Initialization ##########################################

# Latent space dimension (embedding)
input_dim = preprocessing.selection.size

# Model creation with specified input dimension
model = AutoEncoder(input_dim=input_dim).to(device)

# Hyperparameters
N_epochs = 100
batch_size = 2048
learning_rate = 0.0002
alpha = 100

# DataLoaders for batching and shuffling data during training/testing
trainLoader = DataLoader(trainSet, batch_size=batch_size, shuffle=True, num_workers=0)
testLoader = DataLoader(testSet, batch_size=batch_size, shuffle=True, num_workers=0)
testLoader_sig = DataLoader(testSet_sig, batch_size=batch_size, shuffle=True, num_workers=0)

# Loss function definition
loss_function = nn.MSELoss()

# Optimizer setup (Adam optimizer)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#######################################################################################################
######################################## Training Analysis ############################################

# Lists to store loss values during training and evaluation
trainLoss = []
testLoss = []
sigLoss = []

# Run training and evaluation across epochs
for epoch in range(N_epochs):
    print(f"{epoch}")
    print("Training")
    trainLoss.append(train(model, trainLoader, WeightedMSELoss, optimizer, epoch, alpha))
    print("Validating")
    testLoss.append(test(model, testLoader, loss_function, epoch))
    print("Testing Signal")
    sigLoss.append(test(model, testLoader_sig, loss_function, epoch))

# Save model parameters

# Create directories if they don't exist and save the model
folder = f"models/{signal}"
if not os.path.exists(folder):
    os.makedirs(folder)

torch.save(model.state_dict(), f"{folder}/parameters_{(int(pct * 1000) % 100):02d}.pth")

folder = f"figs/{signal}/{(int(pct * 1000) % 100):02d}/training"
if not os.path.exists(folder):
    os.makedirs(folder)

# Plot and save training loss
fig, axes = plt.subplots(figsize=(8, 6))
axes.scatter(range(N_epochs), trainLoss, marker="*", s=10, label='Training Sample')
axes.set_xlabel('N epochs', fontsize=10)
axes.set_ylabel('Loss', fontsize=10)
axes.legend(loc='upper right', fontsize=10)
axes.set_title('Loss during Training', fontsize=14)
fig.savefig(f"{folder}/train_loss.png")

# Plot and save evaluation loss for both test samples and signal
fig, axes = plt.subplots(figsize=(8, 6))
axes.scatter(range(N_epochs), testLoss, marker="^", s=8, label='Test Sample')
axes.scatter(range(N_epochs), sigLoss, marker="o", s=8, label='Signal: ' + signal)
axes.set_xlabel('N epochs', fontsize=10)
axes.set_ylabel('Loss', fontsize=10)
axes.legend(loc='upper right', fontsize=10)
axes.set_title('Loss during Evaluation', fontsize=14)
fig.savefig(f"{folder}/eval_loss.png")
