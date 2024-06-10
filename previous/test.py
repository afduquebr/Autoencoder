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
from sklearn.metrics import roc_curve, auc
from scipy.spatial.distance import jensenshannon

import torch

from autoencoder import AutoEncoder, loss
from previous.main import main, parse_args

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

selection = pd.read_csv("../dijet-selection.csv", header=None).values[:, 0]
smooth_cols = pd.read_csv("../scale-selection.csv", header=None).values[:, 0]

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
elif scale == "standard":
    scaler = StandardScaler()

sample_bkg = bkg[selection] #.sample(frac=1)
sample_sig1 = sig1[selection] #.sample(frac=1)
sample_sig2 = sig2[selection] #.sample(frac=1)

# Concatenate all datasets for the current column to find the global min and max
all_data = pd.concat([sample_bkg, sample_sig1, sample_sig2])

for col in smooth_cols:
    first_positive = all_data[col][all_data[col] > 0].min()
    all_data[col] = np.where(all_data[col] <= 0, first_positive, all_data[col])

all_data[smooth_cols] = all_data[smooth_cols].apply(lambda x: np.log(x))


# Create a MinMaxScaler object with adjusted parameters for the current column
data_scaled = pd.DataFrame(scaler.fit_transform(all_data), columns=selection)

# Apply scaling to each dataset per column
bkg_scaled = data_scaled.iloc[:len(sample_bkg)]
sig1_scaled = data_scaled.iloc[len(sample_bkg):-len(sample_sig2)]
sig2_scaled = data_scaled.iloc[-len(sample_sig2):]

test_scaled = pd.concat([bkg_scaled.iloc[:100000], sig1_scaled.iloc[:300]])

#######################################################################################################
######################################### Data Rescaling ##############################################

test_bkg = torch.from_numpy(bkg_scaled.values).float().to(device)
test_sig1 = torch.from_numpy(sig1_scaled.values).float().to(device)
test_sig2 = torch.from_numpy(sig2_scaled.values).float().to(device)
weights_bkg = torch.from_numpy(weights_bkg[sample_bkg.index]).float().to(device)
mjj_bkg = torch.from_numpy(mjj_bkg[sample_bkg.index]).float().to(device)

test = torch.from_numpy(test_scaled.values).float().to(device)


#######################################################################################################
########################################## Testing Analysis ############################################

# Latent space dimension (embedding)
input_dim = selection.size

# Load Model
model = AutoEncoder(input_dim = input_dim, mid_dim = mid_dim, latent_dim = latent_dim).to(device)
model.load_state_dict(torch.load(f"models/model_parameters_{scale}_{mid_dim}_{latent_dim}.pth", map_location=device))
model.eval()

# Predictions
with torch.no_grad(): # no need to compute gradients here
    predict_bkg = model(test_bkg)
    predict_sig1 = model(test_sig1)
    predict_sig2 = model(test_sig2)

    predict_test = model(test)

# Determine Reconstruction Error

# MSE per feature
loss_bkg = pd.DataFrame(loss(test_bkg, predict_bkg).numpy(), columns=selection)
loss_sig1 = pd.DataFrame(loss(test_sig1, predict_sig1).numpy(), columns=selection)
loss_sig2 = pd.DataFrame(loss(test_sig2, predict_sig2).numpy(), columns=selection)

loss_test = pd.DataFrame(loss(test, predict_test).numpy(), columns=selection)

# Total MSE
loss_bkg_total = loss_bkg.mean(axis=1)
loss_sig1_total = loss_sig1.mean(axis=1)
loss_sig2_total = loss_sig2.mean(axis=1)

loss_test_total = loss_test.mean(axis=1)


# Plot Total Reconstruction Error
nbins = 40
fig, axes = plt.subplots(figsize=(8,6))
# axes.hist([loss_test_total], nbins, range=(0, 0.8), density=1, histtype='step', stacked=True, alpha=1)
axes.hist([loss_bkg_total], nbins, range=(0, 0.8), density=1, histtype='step', label=['Background'], stacked=True, alpha=1)
axes.hist([loss_sig1_total], nbins, range=(0, 0.8), density=1, histtype='step', label=['Signal 1'], stacked=True, alpha=0.9)
axes.hist([loss_sig2_total], nbins, range=(0, 0.8), density=1, histtype='step', label=['Signal 2'], stacked=True, alpha=0.9)
axes.set_xlabel(r"Reconstruction Error")
axes.set_ylabel("Events")
axes.set_xlim(0, 0.8)
axes.legend(loc='upper right')
fig.savefig(f"figs/testing/reconstruction_error_{scale}_{mid_dim}_{latent_dim}.png")

############################################ ROC Curve ##############################################


loss_total1 = pd.concat([loss_bkg_total, loss_sig1_total], axis=0, ignore_index=1)
labels1 = pd.Series([0]*len(loss_bkg_total) + [1]*len(loss_sig1_total))
loss_total1 = pd.DataFrame({'Loss': loss_total1, 'Label': labels1})

fpr1, tpr1, thresholds1 = roc_curve(loss_total1["Label"], loss_total1["Loss"])
roc_auc1 = auc(fpr1, tpr1)


loss_total2 = pd.concat([loss_bkg_total, loss_sig2_total], axis=0, ignore_index=1)
labels2 = pd.Series([0]*len(loss_bkg_total) + [1]*len(loss_sig2_total))
loss_total2 = pd.DataFrame({'Loss': loss_total2, 'Label': labels2})

fpr2, tpr2, thresholds2 = roc_curve(loss_total2["Label"], loss_total2["Loss"])
roc_auc2 = auc(fpr2, tpr2)

# TEST ROC
loss_total3 = loss_test_total.copy()
labels3 = pd.Series([0]*100000 + [1]*300)
loss_total3 = pd.DataFrame({'Loss': loss_total3, 'Label': labels3})

fpr3, tpr3, thresholds3 = roc_curve(loss_total3["Label"], loss_total3["Loss"])
roc_auc3 = auc(fpr3, tpr3)

# Plot ROC curve
fig, axes = plt.subplots(figsize=(8,6))
# axes.plot(fpr3, tpr3, lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc3)
axes.plot(fpr1, tpr1, lw=2, label='Signal 1 ROC curve (AUC = %0.2f)' % roc_auc1)
axes.plot(fpr2, tpr2, lw=2, label='Signal 2 ROC curve (AUC = %0.2f)' % roc_auc2)
axes.plot([0, 1], [0, 1], lw=2, linestyle='--')
axes.set_xlim([0.0, 1.0])
axes.set_ylim([0.0, 1.05])
axes.set_xlabel('False Positive Rate')
axes.set_ylabel('True Positive Rate')
axes.set_title('Receiver Operating Characteristic (ROC) Curve')
axes.legend(loc="lower right")
fig.savefig(f"figs/testing/ROC_{scale}_{mid_dim}_{latent_dim}.png")


############################################ Normalised Mass Distribution  ##############################################

mjj_test = np.concatenate((mjj_bkg[:100000], mjj_sig1[:300]))

# Get all the percentiles
threshold = np.percentile(loss_bkg_total, np.arange(1, 100))
# threshold = np.percentile(loss_test_total, np.arange(1, 100))

# Plot
nbins = 30
fig, axes = plt.subplots(figsize=(8,6))
axes.hist([bkg.mj1j2], nbins, range=(2700, 5000), density=1, histtype='step', label=['No selection'], stacked=True, alpha=1)
axes.hist([bkg.mj1j2[loss_bkg_total > threshold[85 - 1]]], nbins, range=(2700, 5000), density=1, histtype='step', label=['85%'], stacked=True, alpha=0.8)
axes.hist([bkg.mj1j2[loss_bkg_total > threshold[50 - 1]]], nbins, range=(2700, 5000), density=1, histtype='step', label=['50%'], stacked=True, alpha=0.6)
# axes.hist([mjj_test], nbins, range=(2700, 5000), density=1, histtype='step', label=['No selection'], stacked=True, alpha=1)
# axes.hist([mjj_test[loss_test_total > threshold[85 - 1]]], nbins, range=(2700, 5000), density=1, histtype='step', label=['85%'], stacked=True, alpha=0.8)
# axes.hist([mjj_test[loss_test_total > threshold[50 - 1]]], nbins, range=(2700, 5000), density=1, histtype='step', label=['50%'], stacked=True, alpha=0.6)
axes.set_xlabel(r"$m_{jet_1•jet_2}$ [GeV]")
axes.set_ylabel("Events")
axes.set_xlim(2700, 5000)
axes.legend()
fig.savefig(f"figs/testing/mass_dist_{scale}_{mid_dim}_{latent_dim}.png")

############################################ Jensen Shannon Distribution  ##############################################

# Reference uncut histogram
# hist_ref, bins = np.histogram(bkg.mj1j2, bins=30, range=scope)
hist_ref, bins = np.histogram(mjj_test, bins=30, range=scope)

# Loop over percentiles
jsd = []
for th in threshold:
    hist_cut, _ = np.histogram(bkg.mj1j2[loss_bkg_total > th], bins=bins, range=scope)
    # hist_cut, _ = np.histogram(mjj_test[loss_test_total > th], bins=bins, range=scope)
    jsd.append(jensenshannon(hist_cut, hist_ref))

# Plot JS Dist 
fig, axes = plt.subplots(figsize=(8,6))
axes.plot(np.arange(1, 100), jsd, '-', lw=1)
axes.set_xlabel('Percentile Cut')
axes.set_ylabel('JS Distance')
axes.legend()
fig.savefig(f"figs/testing/jd_dist_{scale}_{mid_dim}_{latent_dim}.png")

################################################ Mean Loss per Feature  #################################################

fig, axes = plt.subplots(figsize=(8,6))
axes.bar(range(loss_bkg.columns.size), loss_bkg.mean().values)
# axes.bar(range(loss_test.columns.size), loss_test.mean().values)
axes.set_xlabel("Features")
axes.set_ylabel("Reconstruction error")
axes.set_yscale("log")
fig.savefig(f"figs/testing/error_{scale}_{mid_dim}_{latent_dim}.png")

############################################### Mass vs Loss Distribution  ##############################################

# Make it a 1D histogram
_, bins = np.histogram(mjj_bkg.numpy(), bins=50, range=(2700, 5000))

loss_bkg_avg = []
loss_sig1_avg = []
loss_sig2_avg = []
for i in range(len(bins) - 1):
    loss_bkg_bin = loss_bkg_total[(pd.Series(mjj_bkg.numpy()) >= bins[i]) & (pd.Series(mjj_bkg.numpy()) < bins[i + 1])]
    loss_sig1_bin = loss_sig1_total[(mjj_sig1 >= bins[i]) & (mjj_sig1 < bins[i + 1])]
    loss_sig2_bin = loss_sig2_total[(mjj_sig2 >= bins[i]) & (mjj_sig2 < bins[i + 1])]

    loss_bkg_bin_avg = np.mean(loss_bkg_bin)
    loss_sig1_bin_avg = np.mean(loss_sig1_bin)
    loss_sig2_bin_avg = np.mean(loss_sig2_bin)

    loss_bkg_avg.append(loss_bkg_bin_avg)
    loss_sig1_avg.append(loss_sig1_bin_avg)
    loss_sig2_avg.append(loss_sig2_bin_avg)

# Plot Avg Loss v. Mass
fig, axes = plt.subplots(figsize=(8,6))
axes.plot(bins[:-1] + np.diff(bins) / 2, loss_bkg_avg, label='Background')
axes.plot(bins[:-1] + np.diff(bins) / 2, loss_sig1_avg, label='Signal 1')
axes.plot(bins[:-1] + np.diff(bins) / 2, loss_sig2_avg, label='Signal 2')
axes.set_xlim([2700, 5000])
axes.set_xlabel(r"$m_{jet_1•jet_2}$")
axes.set_ylabel('Reconstruction Error')
axes.set_title('Avg Error v. Mass Distribution')
axes.legend()
fig.savefig(f"figs/testing/AvgLossMass_{scale}_{mid_dim}_{latent_dim}.png")


#########

nbins = 50
fig, axes = plt.subplots(figsize=(8,6))
axes.hist([mjj_bkg], nbins, histtype='step', weights=weights_bkg.cpu().numpy(), label=['Background'], stacked=True, alpha=1)
axes.set_xlabel(r"$m_{jet_1•jet_2}$")
axes.set_ylabel("Events")
axes.legend()
fig.savefig(f"figs/testing/normalised_mass_dist_{scale}_{mid_dim}_{latent_dim}.png")