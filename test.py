#######################################################################################################
"""
Created on Jun 10 2024

@author: Andres Felipe DUQUE BRAN
"""
#######################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc 
from scipy.spatial.distance import jensenshannon 

import torch

from autoencoder import AutoEncoder, loss
from preprocessing import Preprocessor
from  main import parse_args

####################################### GPU or CPU running ###########################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#######################################################################################################
####################################### Data Preprocessing ###########################################

_, signal, pct = parse_args()
preprocessing = Preprocessor()
sample_scaled, bkg_scaled, sig_scaled = preprocessing.get_scaled_data()
weights_sample = preprocessing.get_weights()
mjj_sample, mjj_bkg, mjj_sig = preprocessing.get_mass()

#######################################################################################################
######################################### Data Rescaling ##############################################

test_sample = torch.from_numpy(sample_scaled[:100000].values).float().to(device)
test_bkg = torch.from_numpy(bkg_scaled[:100000].values).float().to(device)
test_sig = torch.from_numpy(sig_scaled.values).float().to(device)

#######################################################################################################
########################################## Testing Analysis ############################################

# Latent space dimension (embedding)
input_dim = preprocessing.selection.size

# Load Model
model = AutoEncoder(input_dim = input_dim).to(device)
model.load_state_dict(torch.load(f"models/model2_parameters_{signal}_{(int(pct * 1000) % 100):02d}.pth", map_location=device))
model.eval()

# Predictions
with torch.no_grad(): # no need to compute gradients here
    predict_sample = model(test_sample)
    predict_bkg = model(test_bkg)
    predict_sig = model(test_sig)

# Determine Reconstruction Error

# MSE per feature
loss_sample = pd.DataFrame(loss(test_sample, predict_sample).numpy(), columns=preprocessing.selection)
loss_bkg = pd.DataFrame(loss(test_bkg, predict_bkg).numpy(), columns=preprocessing.selection)
loss_sig = pd.DataFrame(loss(test_sig, predict_sig).numpy(), columns=preprocessing.selection)

# Total MSE
loss_sample_total = loss_sample.mean(axis=1)
loss_bkg_total = loss_bkg.mean(axis=1)
loss_sig_total = loss_sig.mean(axis=1)

# Plot Total Reconstruction Error
nbins = 40
fig, axes = plt.subplots(figsize=(8,6))
axes.hist([loss_sample_total], nbins, range=(0, 2), density=1, histtype='step', label=['Data'], stacked=True, alpha=1)
axes.hist([loss_bkg_total], nbins, range=(0, 2), density=1, histtype='step', label=['Background'], stacked=True, alpha=0.9)
axes.hist([loss_sig_total], nbins, range=(0, 2), density=1, histtype='step', label=['Signal'], stacked=True, alpha=0.8)
axes.set_xlabel(r"Reconstruction Error")
axes.set_ylabel("Events")
axes.set_xlim(0, 2)
axes.legend(loc='upper right')
fig.savefig(f"figs/testing/reconstruction_error_{signal}_{(int(pct * 1000) % 100):02d}.png")

############################################ ROC Curve ##############################################

# TEMP
####################################################################
# Load Model
model2 = AutoEncoder(input_dim = input_dim).to(device)
model2.load_state_dict(torch.load(f"models/model2_parameters_{signal}_{(int(pct * 1000) % 100):02d}.pth", map_location=device))
model2.eval()

# Predictions
with torch.no_grad(): # no need to compute gradients here
    predict2_sample = model2(test_sample)
    predict2_bkg = model2(test_bkg)
    predict2_sig = model2(test_sig)

# Determine Reconstruction Error

# MSE per feature
loss2_sample = pd.DataFrame(loss(test_sample, predict2_sample).numpy(), columns=preprocessing.selection)
loss2_bkg = pd.DataFrame(loss(test_bkg, predict2_bkg).numpy(), columns=preprocessing.selection)
loss2_sig = pd.DataFrame(loss(test_sig, predict2_sig).numpy(), columns=preprocessing.selection)

# Total MSE
loss2_sample_total = loss_sample.mean(axis=1)
loss2_bkg_total = loss_bkg.mean(axis=1)
loss2_sig_total = loss_sig.mean(axis=1)

loss2_total = pd.concat([loss2_bkg_total, loss2_sig_total], axis=0, ignore_index=1)
labels2 = pd.Series([0]*len(loss2_bkg_total) + [1]*len(loss2_sig_total))
loss2_total = pd.DataFrame({'Loss': loss2_total, 'Label': labels2})

fpr2, tpr2, thresholds2 = roc_curve(loss2_total["Label"], loss2_total["Loss"])
roc_auc2 = auc(fpr2, tpr2)
######################################################

loss_total = pd.concat([loss_bkg_total, loss_sig_total], axis=0, ignore_index=1)
labels = pd.Series([0]*len(loss_bkg_total) + [1]*len(loss_sig_total))
loss_total = pd.DataFrame({'Loss': loss_total, 'Label': labels})

fpr, tpr, thresholds = roc_curve(loss_total["Label"], loss_total["Loss"])
roc_auc = auc(fpr, tpr)


# Plot ROC curve
fig, axes = plt.subplots(figsize=(8,6))
# axes.plot(fpr, tpr, lw=2, label='Signal ROC curve (AUC = %0.2f)' % roc_auc)
axes.plot(fpr, tpr, lw=2, label='First Training (AUC = %0.2f)' % roc_auc)
axes.plot([0, 1], [0, 1], lw=2, linestyle='--')
axes.plot(fpr2, tpr2, lw=2, label='Full Training (AUC = %0.2f)' % roc_auc2)
axes.set_xlim([0.0, 1.0])
axes.set_ylim([0.0, 1.05])
axes.set_xlabel('Signal Efficiency')
axes.set_ylabel('Backgroung Efficiency')
axes.set_title('Receiver Operating Characteristic (ROC) Curve')
axes.legend(loc="lower right")
fig.savefig(f"figs/testing/ROC_{signal}_{(int(pct * 1000) % 100):02d}.png")


############################################ Normalised Mass Distribution  ##############################################

# Get all the percentiles
threshold = np.percentile(loss_sample_total, np.arange(1, 100))

# Plot
nbins = 30
fig, axes = plt.subplots(figsize=(8,6))
axes.hist([mjj_sample[:100000]], nbins, range=(2700, 5000), density=1, histtype='step', label=['No selection'], stacked=True, alpha=0.6)
# axes.hist([mjj_sample[:100000][loss_sample_total > threshold[85 - 1]]], nbins, range=(2700, 5000), density=1, histtype='step', label=['85%'], stacked=True, alpha=1)
axes.hist([mjj_sample[:100000][loss_sample_total > threshold[90 - 1]]], nbins, range=(2700, 5000), density=1, histtype='step', label=['90%'], stacked=True, alpha=0.8)
axes.set_xlabel(r"$m_{jet_1•jet_2}$ [GeV]")
axes.set_ylabel("Events")
axes.set_xlim(2700, 5000)
axes.legend()
fig.savefig(f"figs/testing/mass_dist_{signal}_{(int(pct * 1000) % 100):02d}.png")

############################################ Jensen Shannon Distribution  ##############################################

# Reference uncut histogram
scope = [2700, 5000]
hist_ref, bins = np.histogram(mjj_sample[:100000], bins=30, range=scope)

# Loop over percentiles
jsd = []
for th in threshold:
    hist_cut, _ = np.histogram(mjj_sample[:100000][loss_sample_total > th], bins=bins, range=scope)
    jsd.append(jensenshannon(hist_cut, hist_ref))


df = pd.DataFrame(jsd)
df.to_csv(f"figs/testing/Jensen Shannon/jsd.csv", index=False, header=False)

# Plot JS Dist 
fig, axes = plt.subplots(figsize=(8,6))
axes.plot(np.arange(1, 100), jsd, '-', lw=1)
axes.set_xlabel('Percentile Cut')
axes.set_ylabel('JS Distance')
# axes.legend()
fig.savefig(f"figs/testing/jd_dist_{signal}_{(int(pct * 1000) % 100):02d}.png")

################################################ Mean Loss per Feature  #################################################

fig, axes = plt.subplots(figsize=(8,6))
axes.bar(range(loss_sample.columns.size), loss_sample.mean().values)
axes.set_xlabel("Features")
axes.set_ylabel("Reconstruction error")
axes.set_yscale("log")
fig.savefig(f"figs/testing/error_{signal}_{(int(pct * 1000) % 100):02d}.png")

############################################### Mass vs Loss Distribution  ##############################################

# Make it a 1D histogram
_, bins = np.histogram(mjj_sample[:100000], bins=50, range=(2700, 5000))

loss_sample_avg = []
loss_bkg_avg = []
loss_sig_avg = []

for i in range(len(bins) - 1):
    loss_sample_bin = loss_sample_total[(mjj_sample[:100000] >= bins[i]) & (mjj_sample[:100000] < bins[i + 1])]
    loss_bkg_bin = loss_bkg_total[(mjj_bkg[:100000] >= bins[i]) & (mjj_bkg[:100000] < bins[i + 1])]
    loss_sig_bin = loss_sig_total[(mjj_sig >= bins[i]) & (mjj_sig < bins[i + 1])]

    loss_sample_bin_avg = np.mean(loss_sample_bin)
    loss_bkg_bin_avg = np.mean(loss_bkg_bin)
    loss_sig_bin_avg = np.mean(loss_sig_bin)

    loss_sample_avg.append(loss_sample_bin_avg)
    loss_bkg_avg.append(loss_bkg_bin_avg)
    loss_sig_avg.append(loss_sig_bin_avg)

# Plot Avg Loss v. Mass
fig, axes = plt.subplots(figsize=(8,6))
axes.plot(bins[:-1] + np.diff(bins) / 2, loss_sample_avg, label='Data')
axes.plot(bins[:-1] + np.diff(bins) / 2, loss_bkg_avg, label='Background')
axes.plot(bins[:-1] + np.diff(bins) / 2, loss_sig_avg, label='Signal')
axes.set_xlim([2700, 5000])
axes.set_xlabel(r"$m_{jet_1•jet_2}$")
axes.set_ylabel('Reconstruction Error')
axes.set_title('Avg Error v. Mass Distribution')
axes.legend()
fig.savefig(f"figs/testing/AvgLossMass_{signal}_{(int(pct * 1000) % 100):02d}.png")