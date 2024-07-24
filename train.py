#######################################################################################################
"""
Created on Jun 10 2024

@author: Andres Felipe DUQUE BRAN
"""
#######################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from autoencoder import AutoEncoder, WeightedMSELoss, train, test
from preprocessing import Preprocessor

####################################### GPU or CPU running ###########################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#######################################################################################################
####################################### Data Preprocessing ###########################################

preprocessing = Preprocessor()
sample_scaled, _, sig_scaled = preprocessing.get_scaled_data()
weights_sample = preprocessing.get_weights()
mjj_sample, _, _ = preprocessing.get_mass()

#######################################################################################################
######################################## Data in Tensors ###########################################

train_sample = torch.from_numpy(sample_scaled[:100000].values).float().to(device)
test_sample = torch.from_numpy(sample_scaled[100000:200000].values).float().to(device)
test_sig = torch.from_numpy(sig_scaled.values).float().to(device)
weights = torch.from_numpy(weights_sample[:100000]).float().to(device)
mjj = torch.from_numpy(mjj_sample[:100000]).float().to(device)

trainSet = TensorDataset(train_sample, weights, mjj)
testSet = TensorDataset(test_sample, test_sample)
testSet_sig = TensorDataset(test_sig, test_sig)

#######################################################################################################
######################################### Model Initlization ##########################################

# Latent space dimension (embedding)
input_dim = preprocessing.selection.size

# Model creation
model = AutoEncoder(input_dim = input_dim).to(device)

# Hyperparameters
N_epochs = 100
batch_size = 2048
learning_rate = 0.0002
alpha = 100

# dataloaders
trainLoader = DataLoader(trainSet, batch_size=batch_size, shuffle=True, num_workers=4)
testLoader = DataLoader(testSet, batch_size=batch_size, shuffle=True, num_workers=4)
testLoader_sig = DataLoader(testSet_sig, batch_size=batch_size, shuffle=True, num_workers=4)

# Loss function
loss_function = nn.MSELoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

#######################################################################################################
######################################## Training Analysis ############################################

trainLoss = []
testLoss = []
sigLoss = []

# Run training and store validation through training
for epoch in range(N_epochs) :
    print(f"{epoch}")
    print("Training")
    trainLoss.append(train(model, trainLoader, WeightedMSELoss, optimizer, epoch, alpha))
    print("Validating")
    testLoss.append(test(model, testLoader, loss_function, epoch))
    print("Testing Signal")
    sigLoss.append(test(model, testLoader_sig, loss_function, epoch))

# Save model
torch.save(model.state_dict(), f"models/model_parameters_{preprocessing.signal}_{(int(preprocessing.pct * 1000) % 100):02d}.pth")

# Create Loss per Epochs
fig, axes = plt.subplots(figsize=(8,6))
axes.scatter(range(N_epochs), trainLoss, marker="*", s=10, label='Training Sample')
axes.set_xlabel('N epochs',fontsize=10)
axes.set_ylabel('Loss',fontsize=10)
axes.legend(loc='upper right',fontsize=10)
axes.set_title('Loss during Training',fontsize=14)
fig.savefig(f"figs/training/train_loss_{preprocessing.signal}_{(int(preprocessing.pct * 1000) % 100):02d}.png")

# Create Loss per Epochs
fig, axes = plt.subplots(figsize=(8,6))
axes.scatter(range(N_epochs), testLoss, marker="^", s=8, label='Test Sample')
axes.scatter(range(N_epochs), sigLoss, marker="o", s=8, label='Signal: ' + preprocessing.signal)
axes.set_xlabel('N epochs',fontsize=10)
axes.set_ylabel('Loss',fontsize=10)
axes.legend(loc='upper right',fontsize=10)
axes.set_title('Loss during Evaluation',fontsize=14)
fig.savefig(f"figs/training/eval_loss_{preprocessing.signal}_{(int(preprocessing.pct * 1000) % 100):02d}.png")