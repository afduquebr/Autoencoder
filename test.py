#######################################################################################################
"""
Created on Jun 10 2024

@author: Andres Felipe DUQUE BRAN
"""
#######################################################################################################

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc
from scipy.spatial.distance import jensenshannon

import torch

from autoencoder import AutoEncoder, loss
from preprocessing import Preprocessor
from main import parse_args

####################################### GPU or CPU running ###########################################

# Determine whether to use a GPU or CPU for computations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#######################################################################################################
####################################### Data Preprocessing ###########################################

# Parse arguments to get signal and anomaly percentage
signal, pct = parse_args()

# Initialize data preprocessing
preprocessing = Preprocessor()

# Retrieve scaled datasets (sample, background, and signal)
sample_scaled, bkg_scaled, sig_scaled = preprocessing.get_scaled_data()

# Retrieve weights and mass distributions
weights_sample = preprocessing.get_weights()
mjj_sample, mjj_bkg, mjj_sig = preprocessing.get_mass()

#######################################################################################################
######################################### Data Rescaling ##############################################

# Convert numpy arrays to torch tensors and move them to the appropriate device
test_sample = torch.from_numpy(sample_scaled[:100000].values).float().to(device)
test_bkg = torch.from_numpy(bkg_scaled[:100000].values).float().to(device)
test_sig = torch.from_numpy(sig_scaled[:100000].values).float().to(device)

#######################################################################################################
########################################## Testing Analysis ###########################################

# Latent space dimension (embedding)
input_dim = preprocessing.selection.size

# Load pre-trained AutoEncoder model
model = AutoEncoder(input_dim=input_dim).to(device)
model.load_state_dict(torch.load(f"models/{signal}/parameters_{(int(pct * 1000) % 100):02d}.pth", map_location=device))
model.eval()

# Make predictions using the model
with torch.no_grad():  # No need to compute gradients during inference
    predict_sample = model(test_sample)
    predict_bkg = model(test_bkg)
    predict_sig = model(test_sig)

# Calculate Reconstruction Error (MSE per feature)
loss_sample = pd.DataFrame(loss(test_sample, predict_sample).numpy(), columns=preprocessing.selection)
loss_bkg = pd.DataFrame(loss(test_bkg, predict_bkg).numpy(), columns=preprocessing.selection)
loss_sig = pd.DataFrame(loss(test_sig, predict_sig).numpy(), columns=preprocessing.selection)

# Calculate total MSE by averaging across features
loss_sample_total = loss_sample.mean(axis=1)
loss_bkg_total = loss_bkg.mean(axis=1)
loss_sig_total = loss_sig.mean(axis=1)

# Create directory for saving figures
folder = f"figs/{signal}/{(int(pct * 1000) % 100):02d}/testing"
if not os.path.exists(folder):
    os.makedirs(folder)

# Plot Total Reconstruction Error
nbins = 40
fig, axes = plt.subplots(figsize=(8,6))
axes.hist([loss_sample_total], nbins, range=(0, 2), density=1, histtype='step', label=['Data'], stacked=True, alpha=1)
axes.hist([loss_bkg_total], nbins, range=(0, 2), density=1, histtype='step', label=['Background'], stacked=True, alpha=0.9)
axes.hist([loss_sig_total], nbins, range=(0, 2), density=1, histtype='step', label=['Signal'], stacked=True, alpha=0.8)
axes.set_xlabel("Reconstruction Error")
axes.set_ylabel("Events")
axes.set_xlim(0, 2)
axes.legend(loc='upper right')
fig.savefig(f"{folder}/avg_reconstruction_error.png")

############################################ ROC Curve ##############################################

# Combine losses and labels for ROC curve calculation
loss_total = pd.concat([loss_bkg_total, loss_sig_total], axis=0, ignore_index=1)
labels = pd.Series([0]*len(loss_bkg_total) + [1]*len(loss_sig_total))
loss_total = pd.DataFrame({'Loss': loss_total, 'Label': labels})

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(loss_total["Label"], loss_total["Loss"])
roc_auc = auc(fpr, tpr)

# Plot ROC curve
fig, axes = plt.subplots(figsize=(8,6))
axes.plot(fpr, tpr, lw=2, label='Signal ROC curve (AUC = %0.2f)' % roc_auc)
axes.plot([0, 1], [0, 1], lw=2, linestyle='--')
axes.set_xlim([0.0, 1.0])
axes.set_ylim([0.0, 1.05])
axes.set_xlabel('Signal Efficiency')
axes.set_ylabel('Background Efficiency')
axes.set_title('Receiver Operating Characteristic (ROC) Curve')
axes.legend(loc="lower right")
fig.savefig(f"{folder}/ROC.png")

############################################ Normalized Mass Distribution ##############################################

# Get all the percentiles of reconstruction error
threshold = np.percentile(loss_sample_total, np.arange(1, 100))

# Plot the normalized mass distribution with and without anomaly cut
nbins = 30
fig, axes = plt.subplots(figsize=(8,6))
axes.hist([mjj_sample[:100000]], nbins, range=(2700, 5000), density=1, histtype='step', label=['No selection'], stacked=True, alpha=0.6)
axes.hist([mjj_sample[:100000][loss_sample_total > threshold[90 - 1]]], nbins, range=(2700, 5000), density=1, histtype='step', label=['90%'], stacked=True, alpha=0.8)
axes.set_xlabel(r"$m_{jet_1\cdot jet_2}$ [GeV]")
axes.set_ylabel("Events")
axes.set_xlim(2700, 5000)
axes.legend()
fig.savefig(f"{folder}/mass_dist.png")

############################################ Jensen-Shannon Distribution ##############################################

# Reference uncut histogram
scope = [2700, 5000]
hist_ref, bins = np.histogram(mjj_sample[:100000], bins=30, range=scope)

# Calculate Jensen-Shannon Divergence for different thresholds
jsd = []
for th in threshold:
    hist_cut, _ = np.histogram(mjj_sample[:100000][loss_sample_total > th], bins=bins, range=scope)
    jsd.append(jensenshannon(hist_cut, hist_ref))

# Plot Jensen-Shannon Divergence
fig, axes = plt.subplots(figsize=(8,6))
axes.plot(np.arange(1, 100), jsd, '-', lw=1)
axes.set_xlabel('Percentile Cut')
axes.set_ylabel('JS Distance')
fig.savefig(f"{folder}/jd_dist.png")

################################################ Mean Loss per Feature #################################################

# Plot Mean Reconstruction Error per Feature
fig, axes = plt.subplots(figsize=(8,6))
axes.bar(range(loss_sample.columns.size), loss_sample.mean().values)
axes.set_xlabel("Features")
axes.set_ylabel("Reconstruction Error")
axes.set_yscale("log")
fig.savefig(f"{folder}/reconstruction_error_features.png")

############################################### Mass vs Loss Distribution ##############################################

# Calculate average reconstruction error per mass bin
_, bins = np.histogram(mjj_sample[:100000], bins=50, range=(2700, 5000))
loss_sample_avg = []
loss_bkg_avg = []
loss_sig_avg = []

for i in range(len(bins) - 1):
    # Get the mean loss for each bin
    loss_sample_bin = loss_sample_total[(mjj_sample[:100000] >= bins[i]) & (mjj_sample[:100000] < bins[i + 1])]
    loss_bkg_bin = loss_bkg_total[(mjj_bkg[:100000] >= bins[i]) & (mjj_bkg[:100000] < bins[i + 1])]
    loss_sig_bin = loss_sig_total[(mjj_sig >= bins[i]) & (mjj_sig < bins[i + 1])]

    loss_sample_avg.append(np.mean(loss_sample_bin))
    loss_bkg_avg.append(np.mean(loss_bkg_bin))
    loss_sig_avg.append(np.mean(loss_sig_bin))

# Plot Average Loss vs. Mass Distribution
fig, axes = plt.subplots(figsize=(8,6))
axes.plot(bins[:-1] + np.diff(bins) / 2, loss_sample_avg, label='Data')
axes.plot(bins[:-1] + np.diff(bins) / 2, loss_bkg_avg, label='Background')
axes.plot(bins[:-1] + np.diff(bins) / 2, loss_sig_avg, label='Signal')
axes.set_xlim([2700, 5000])
axes.set_xlabel(r"$m_{jet_1\cdot jet_2}$")
axes.set_ylabel('Reconstruction Error')
axes.set_title('Avg Error vs. Mass Distribution')
axes.legend()
fig.savefig(f"{folder}/AvgLossMass.png")
