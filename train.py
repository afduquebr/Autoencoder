#######################################################################################################
"""
Created on Jun 10 2024

@author: Andres Felipe DUQUE BRAN
"""
#######################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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

path, signal, pct = parse_args()
main()

if path == "local":
    path = "../GAN-AE/clustering-lhco/data"
elif path == "server": 
    path = "/AtlasDisk/user/duquebran/clustering-lhco/data"

bkg = pd.read_hdf(f"{path}/RnD_2j_scalars_bkg.h5")
sig1 = pd.read_hdf(f"{path}/RnD_2j_scalars_sig.h5")
sig2 = pd.read_hdf(f"{path}/RnD2_2j_scalars_sig.h5")
bbox = pd.read_hdf(f"{path}/BBOX1_2j_scalars_sig.h5")

selection = pd.read_csv("dijet-selection.csv", header=None).values[:, 0]
smooth_cols = pd.read_csv("scale-selection.csv", header=None).values[:, 0]

bkg.replace([np.nan, -np.inf, np.inf], 0, inplace=True)
sig1.replace([np.nan, -np.inf, np.inf], 0, inplace=True)
sig2.replace([np.nan, -np.inf, np.inf], 0, inplace=True)
bbox.replace([np.nan, -np.inf, np.inf], 0, inplace=True)

mass = 'mj1j2'
scope = [2700, 5000]

bkg = bkg[(bkg[mass] > scope[0]) & (bkg[mass] < scope[1])].reset_index()
sig1 = sig1[(sig1[mass] > scope[0]) & (sig1[mass] < scope[1])].reset_index()
sig2 = sig2[(sig2[mass] > scope[0]) & (sig2[mass] < scope[1])].reset_index()
bbox = bbox[(bbox[mass] > scope[0]) & (bbox[mass] < scope[1])].reset_index()

# Mix signal or bbox with bkg

if signal != None:
    sample_sig = globals()[signal].sample(frac=1)
    sample = pd.concat([bkg, sample_sig[:int(pct * len(bkg))]]).sample(frac=1)
else:
    signal = "sig1"
    pct = 0
    sample_sig = sig1.sample(frac=1)
    sample = bkg.sample(frac=1)

mjj_sample = sample[mass].values
mjj_sig = sample_sig[mass].values

#######################################################################################################
############################################# Reweighting #############################################

Hc,Hb = np.histogram(mjj_sample,bins=500)
weights = np.array(Hc,dtype=float)
weights[weights > 0.0] = 1.0 / weights[weights > 0.0]
weights[weights == 0.0] = 1.0
weights = np.append(weights, weights[-1])
weights *= 1000.0 # To avoid very small weights
weights = weights[np.searchsorted(Hb, mjj_sample)]

#######################################################################################################
######################################## Data Preprocessing ###########################################

# Concatenate all datasets for the current column to find the global min and max
all_data = pd.concat([sample[selection], sample_sig[selection]])

for col in smooth_cols:
    first_positive = all_data[col][all_data[col] > 0].min()
    all_data[col] = np.where(all_data[col] <= 0, first_positive, all_data[col])

all_data[smooth_cols] = all_data[smooth_cols].apply(lambda x: np.log(x))

# Create a Scaler object with adjusted parameters for each column
scaler = StandardScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(all_data), columns=selection)

# Apply scaling to each dataset per column
sample_scaled = data_scaled.iloc[:len(sample)]
sig_scaled = data_scaled.iloc[len(sample):]

#######################################################################################################
######################################## Data in Tensors ###########################################

train_sample = torch.from_numpy(sample_scaled[:100000].values).float().to(device)
test_sample = torch.from_numpy(sample_scaled[100000:200000].values).float().to(device)
test_sig = torch.from_numpy(sig_scaled.values).float().to(device)
weights = torch.from_numpy(weights[:100000]).float().to(device)
mjj = torch.from_numpy(mjj_sample[:100000]).float().to(device)

trainSet = TensorDataset(train_sample, weights, mjj)
testSet = TensorDataset(test_sample, test_sample)
testSet_sig = TensorDataset(test_sig, test_sig)

#######################################################################################################
######################################### Model Initlization ##########################################

# Latent space dimension (embedding)
input_dim = selection.size

# Model creation
model = AutoEncoder(input_dim = input_dim).to(device)

# Hyperparameters
N_epochs = 100
batch_size = 2048
learning_rate = 0.0002
alpha = 50

# dataloaders
trainLoader = DataLoader(trainSet, batch_size=batch_size, shuffle=True, num_workers=0)
testLoader = DataLoader(testSet, batch_size=batch_size, shuffle=True, num_workers=0)
testLoader_sig = DataLoader(testSet_sig, batch_size=batch_size, shuffle=True, num_workers=0)

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
    print("Training")
    trainLoss.append(train(model, trainLoader, WeightedMSELoss, optimizer, epoch, alpha))
    print("Validating")
    testLoss.append(test(model, testLoader, loss_function, epoch))
    print("Testing Signal")
    sigLoss.append(test(model, testLoader_sig, loss_function, epoch))

# Save model
torch.save(model.state_dict(), f"models/model_parameters_{signal}_{(int(pct * 1000) % 100):02d}.pth")

# Create Loss per Epochs
fig, axes = plt.subplots(figsize=(8,6))
axes.scatter(range(N_epochs), trainLoss, marker="*", s=10, label='Training Sample')
axes.set_xlabel('N epochs',fontsize=10)
axes.set_ylabel('Loss',fontsize=10)
axes.legend(loc='upper right',fontsize=10)
axes.set_title('Loss during Training',fontsize=14)
fig.savefig(f"figs/training/train_loss_{signal}_{(int(pct * 1000) % 100):02d}.png")

# Create Loss per Epochs
fig, axes = plt.subplots(figsize=(8,6))
axes.scatter(range(N_epochs), testLoss, marker="^", s=8, label='Test Sample')
axes.scatter(range(N_epochs), sigLoss, marker="o", s=8, label='Signal: ' + signal)
axes.set_xlabel('N epochs',fontsize=10)
axes.set_ylabel('Loss',fontsize=10)
axes.legend(loc='upper right',fontsize=10)
axes.set_title('Loss during Evaluation',fontsize=14)
fig.savefig(f"figs/training/eval_loss_{signal}_{(int(pct * 1000) % 100):02d}.png")