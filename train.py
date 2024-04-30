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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from autoencoder import AutoEncoder, WeightedMSELoss, train, test
from main import main, parse_args

####################################### GPU or CPU running ###########################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#######################################################################################################
####################################### Data Initialization ###########################################

path, scale, mid_dim, latent_dim = parse_args()
main()

if path == "local":
    path = "../GAN-AE/clustering-lhco/data"
elif path == "server": 
    path = "/AtlasDisk/user/duquebran/clustering-lhco/data"

bkg = pd.read_hdf(f"{path}/RnD_2j_scalars_bkg.h5")
sig1 = pd.read_hdf(f"{path}/RnD_2j_scalars_sig.h5")
sig2 = pd.read_hdf(f"{path}/RnD2_2j_scalars_sig.h5")

selection = pd.read_csv(f"dijet-selection.csv", header=None).values[:, 0]

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

#######################################################################################################
######################################## Data Preprocessing ###########################################

if scale == "minmax":
    scaler = MinMaxScaler()
elif scale == "standard":
    scaler = StandardScaler()

sample_bkg = bkg[selection].sample(frac=1)
sample_sig1 = sig1[selection].sample(frac=1)
sample_sig2 = sig2[selection].sample(frac=1)

bkg_scaled = pd.DataFrame(scaler.fit_transform(sample_bkg), columns=selection)
sig1_scaled = pd.DataFrame(scaler.transform(sample_sig1), columns=selection)
sig2_scaled = pd.DataFrame(scaler.transform(sample_sig2), columns=selection)
train_bkg = bkg_scaled[(sig1_scaled.shape[0]):]
test_bkg = bkg_scaled[:(sig2_scaled.shape[0])]

train_bkg = torch.from_numpy(train_bkg.values).float().to(device)
test_bkg = torch.from_numpy(test_bkg.values).float().to(device)
test_sig1 = torch.from_numpy(sig1_scaled.values).float().to(device)
test_sig2 = torch.from_numpy(sig2_scaled.values).float().to(device)
weights_bkg = torch.from_numpy(weights_bkg[sample_bkg.index][sig1_scaled.shape[0]:]).float().to(device)
mjj_bkg = torch.from_numpy(mjj_bkg[sample_bkg.index][sig1_scaled.shape[0]:]).float().to(device)

trainSet = TensorDataset(train_bkg, weights_bkg, mjj_bkg)
testSet_bkg = TensorDataset(test_bkg, test_bkg)
testSet_sig1 = TensorDataset(test_sig1, test_sig1)
testSet_sig2 = TensorDataset(test_sig2, test_sig2)

#######################################################################################################
######################################### Model Initlization ##########################################

# Latent space dimension (embedding)
input_dim = selection.size

# Model creation
model = AutoEncoder(input_dim = input_dim, mid_dim = mid_dim, latent_dim = latent_dim).to(device)

# Hyperparameters
N_epochs = 100 #100
batch_size = 2048
learning_rate = 0.0002
alpha = 0

# dataloaders
trainLoader = DataLoader(trainSet, batch_size=batch_size, shuffle=True, num_workers=0)
testLoader_bkg = DataLoader(testSet_bkg, batch_size=batch_size, shuffle=True, num_workers=0)
testLoader_sig1 = DataLoader(testSet_sig1, batch_size=batch_size, shuffle=True, num_workers=0)
testLoader_sig2 = DataLoader(testSet_sig2, batch_size=batch_size, shuffle=True, num_workers=0)

# Loss function
# criterion = WeightedMSELoss
loss_function = nn.MSELoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

#######################################################################################################
######################################## Training Analysis ############################################

trainLoss = []
testLoss = []
sig1Loss = []
sig2Loss = []

# Run training and store validation through training
for epoch in range(N_epochs) :
    print("Training")
    trainLoss.append(train(model, trainLoader, WeightedMSELoss, optimizer, epoch, alpha))
    print("Validating")
    testLoss.append(test(model, testLoader_bkg, loss_function, epoch))
    print("Testing Signal 1")
    sig1Loss.append(test(model, testLoader_sig1, loss_function, epoch))
    print("Testing Signal 2")
    sig2Loss.append(test(model, testLoader_sig2, loss_function, epoch))

# Save model
torch.save(model.state_dict(), f"models/model_parameters_{scale}_{mid_dim}_{latent_dim}.pth")

# Create Loss per Epochs
fig, axes = plt.subplots(figsize=(8,6))
axes.scatter(range(N_epochs), trainLoss, marker="*", s=10, label='Training loss function')
axes.scatter(range(N_epochs), testLoss, marker="^", s=8, label='Background testing loss function')
axes.scatter(range(N_epochs), sig1Loss, marker="o", s=8, label='Signal 1 testing loss function')
axes.scatter(range(N_epochs), sig2Loss, marker="o", s=8, label='Signal 2 testing loss function')
axes.set_xlabel('N epochs',fontsize=10)
axes.set_ylabel('Loss',fontsize=10)
axes.legend(loc='upper right',fontsize=10)
fig.savefig(f"figs/training/loss_{scale}_{mid_dim}_{latent_dim}.png")