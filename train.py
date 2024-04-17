#######################################################################################################
"""
Created on Apr 16 2024

@author: Andres Felipe DUQUE BRAN
"""
#######################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler 
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_predict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision

from autoencoder import AutoEncoder
from test import test

#######################################################################################################
####################################### Data Initialization ###########################################

# data = "../GAN-AE/clustering-lhco/data"
data = "/AtlasDisk/user/duquebran/clustering-lhco/data"

# code = ""
code = "/AtlasDisk/home2/duquebran/Autoencoder/"

scale = "minmax" 
# scale = "standard"

bkg = pd.read_hdf(f"{data}/RnD_2j_scalars_bkg.h5")
sig1 = pd.read_hdf(f"{data}/RnD_2j_scalars_sig.h5")
sig2 = pd.read_hdf(f"{data}/RnD2_2j_scalars_sig.h5")

selection = pd.read_csv(f"{code}dijet-selection.csv", header=None).values[:, 0]

bkg.replace([np.nan, -np.inf, np.inf], 0, inplace=True)
sig1.replace([np.nan, -np.inf, np.inf], 0, inplace=True)
sig2.replace([np.nan, -np.inf, np.inf], 0, inplace=True)

mass = 'mj1j2'
scope = [2700, 5000]

bkg = bkg[(bkg[mass] > scope[0]) & (bkg[mass] < scope[1])].reset_index()
sig1 = sig1[(sig1[mass] > scope[0]) & (sig1[mass] < scope[1])].reset_index()
sig2 = sig2[(sig2[mass] > scope[0]) & (sig2[mass] < scope[1])].reset_index()

mjj_bkg = bkg[mass].values
mjj_sig1 = sig1[mass].values
mjj_sig2 = sig2[mass].values

#######################################################################################################
############################################# Reweighting #############################################

Hc,Hb = np.histogram(mjj_bkg,bins=500)
weights = np.array(Hc,dtype=float)
weights[weights > 0.0] = 1.0 / weights[weights > 0.0]
weights[weights == 0.0] = 1.0
weights = np.append(weights, weights[-1])
weights *= 1000.0 # To avoid very small weights
weights_bkg = weights[np.searchsorted(Hb, mjj_bkg)]
weights_sig1 = weights[np.searchsorted(Hb, mjj_sig1)]
weights_sig2 = weights[np.searchsorted(Hb, mjj_sig2)]

#######################################################################################################
######################################## Data Preprocessing ###########################################

if scale == "minmax":
    scaler = MinMaxScaler()
else:
    scaler = StandardScaler()

bkg_scaled = pd.DataFrame(scaler.fit_transform(bkg[selection].sample(frac=1)), columns=selection).mul(weights_bkg, axis = 0)
sig1_scaled = pd.DataFrame(scaler.transform(sig1[selection].sample(frac=1)), columns=selection).mul(weights_sig1, axis = 0)
sig2_scaled = pd.DataFrame(scaler.transform(sig2[selection].sample(frac=1)), columns=selection).mul(weights_sig2, axis = 0)
train_bkg = bkg_scaled[(sig1_scaled.shape[0]):]
test_bkg = bkg_scaled[:(sig2_scaled.shape[0])]

train_bkg = torch.from_numpy(train_bkg.values).float()
test_bkg = torch.from_numpy(test_bkg.values).float()
test_sig1 = torch.from_numpy(sig1_scaled.values).float()
test_sig2 = torch.from_numpy(sig2_scaled.values).float()

trainSet = TensorDataset(train_bkg, train_bkg)
testSet_bkg = TensorDataset(test_bkg, test_bkg)
testSet_sig1 = TensorDataset(test_sig1, test_sig1)
testSet_sig2 = TensorDataset(test_sig2, test_sig2)


#######################################################################################################
######################################### Model Initlization ##########################################

# Latent space dimension (embedding)
input_dim = selection.size

# Model creation
model = AutoEncoder(input_dim=input_dim)

# Hyperparameters
N_epochs = 100 #100
batch_size = 2048
learning_rate = 0.0002

# dataloaders
trainLoader = DataLoader(trainSet, batch_size=batch_size, shuffle=True, num_workers=0)
testLoader_bkg = DataLoader(testSet_bkg, batch_size=batch_size, shuffle=True, num_workers=0)
testLoader_sig1 = DataLoader(testSet_sig1, batch_size=batch_size, shuffle=True, num_workers=0)
testLoader_sig2 = DataLoader(testSet_sig2, batch_size=batch_size, shuffle=True, num_workers=0)

# Loss function
loss_function = nn.MSELoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

#######################################################################################################
########################################## Model Training #############################################

def train(model, data_loader, opt, epoch):
    model.train()
    for i, (features, _) in enumerate(data_loader):     
        prediction = model(features)
        loss = loss_function(prediction, features)
        opt.zero_grad()
        loss.backward()
        opt.step()

        # print statistics
        if i % 100 == 99:    
            print('[Epoch : %d, iteration: %5d]'% (epoch + 1, (i + 1) + epoch * len(data_loader.dataset)))
            print('Training loss: %.3f'% (loss.item()))
    return loss.item()

#######################################################################################################
######################################## Training Analysis ############################################

trainLoss = []
testLoss = []
sig1Loss = []
sig2Loss = []

# Run training and store validation through training
for epoch in range(N_epochs) :
    print("Training")
    trainLoss.append(train(model, trainLoader, optimizer, epoch))
    print("Validating")
    testLoss.append(test(model, testLoader_bkg, epoch))
    print("Testing Signal 1")
    sig1Loss.append(test(model, testLoader_sig1, epoch))
    print("Testing Signal 2")
    sig2Loss.append(test(model, testLoader_sig2, epoch))

# Save model
torch.save(model.state_dict(), f"{code}models/model_parameters_{scale}.pth")

# Create Loss per Epochs
fig, axes = plt.subplots(figsize=(8,6))
axes.scatter(range(N_epochs), trainLoss, marker="*", s=10, label='Training loss function')
axes.scatter(range(N_epochs), testLoss, marker="^", s=8, label='Background testing loss function')
axes.scatter(range(N_epochs), sig1Loss, marker="o", s=8, label='Signal 1 testing loss function')
axes.scatter(range(N_epochs), sig2Loss, marker="o", s=8, label='Signal 2 testing loss function')
axes.set_xlabel('N epochs',fontsize=10)
axes.set_ylabel('Loss',fontsize=10)
axes.legend(loc='upper right',fontsize=10)
fig.save(f"{code}figs/training/loss_{scale}.png")