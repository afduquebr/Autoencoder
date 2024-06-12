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
from sklearn.metrics import roc_curve, auc
from scipy.spatial.distance import jensenshannon

import torch

from autoencoder import AutoEncoder, loss
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
######################################### Data Rescaling ##############################################

test_sample = torch.from_numpy(sample_scaled[:100000].values).float().to(device)
test_sig = torch.from_numpy(sig_scaled.values).float().to(device)
weights = torch.from_numpy(weights[:100000]).float().to(device)
mjj = torch.from_numpy(mjj_sample[:100000]).float().to(device)

#######################################################################################################
########################################## Testing Analysis ############################################

# Latent space dimension (embedding)
input_dim = selection.size

# Load Model
model = AutoEncoder(input_dim = input_dim).to(device)
model.load_state_dict(torch.load(f"models/model_parameters_{signal}_{(int(pct * 1000) % 100):02d}.pth", map_location=device))
model.eval()

# Predictions
with torch.no_grad(): # no need to compute gradients here
    predict_sample = model(test_sample)
    predict_sig = model(test_sig)

# Determine Reconstruction Error

# MSE per feature
loss_sample = pd.DataFrame(loss(test_sample, predict_sample).numpy(), columns=selection)
loss_sig = pd.DataFrame(loss(test_sig, predict_sig).numpy(), columns=selection)

# Total MSE
loss_sample_total = loss_sample.mean(axis=1)
loss_sig_total = loss_sig.mean(axis=1)

# Plot Total Reconstruction Error
nbins = 40
fig, axes = plt.subplots(figsize=(8,6))
axes.hist([loss_sample_total], nbins, range=(0, 1.5), density=1, histtype='step', label=['Background'], stacked=True, alpha=1)
axes.hist([loss_sig_total], nbins, range=(0, 1.5), density=1, histtype='step', label=['Signal: ' + signal], stacked=True, alpha=0.9)
axes.set_xlabel(r"Reconstruction Error")
axes.set_ylabel("Events")
axes.set_xlim(0, 1.5)
axes.legend(loc='upper right')
fig.savefig(f"figs/testing/reconstruction_error_{signal}_{(int(pct * 1000) % 100):02d}.png")

############################################ ROC Curve ##############################################


loss_total = pd.concat([loss_sample_total, loss_sig_total], axis=0, ignore_index=1)
labels = pd.Series([0]*len(loss_sample_total) + [1]*len(loss_sig_total))
loss_total = pd.DataFrame({'Loss': loss_total, 'Label': labels})

fpr, tpr, thresholds = roc_curve(loss_total["Label"], loss_total["Loss"])
roc_auc = auc(fpr, tpr)


# Plot ROC curve
fig, axes = plt.subplots(figsize=(8,6))
axes.plot(fpr, tpr, lw=2, label='Signal ROC curve (AUC = %0.2f)' % roc_auc)
axes.plot([0, 1], [0, 1], lw=2, linestyle='--')
axes.set_xlim([0.0, 1.0])
axes.set_ylim([0.0, 1.05])
axes.set_xlabel('False Positive Rate')
axes.set_ylabel('True Positive Rate')
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
axes.hist([mjj_sample[:100000][loss_sample_total > threshold[50 - 1]]], nbins, range=(2700, 5000), density=1, histtype='step', label=['50%'], stacked=True, alpha=0.8)
axes.hist([mjj_sample[:100000][loss_sample_total > threshold[99 - 1]]], nbins, range=(2700, 5000), density=1, histtype='step', label=['99%'], stacked=True, alpha=1)
axes.set_xlabel(r"$m_{jet_1•jet_2}$ [GeV]")
axes.set_ylabel("Events")
axes.set_xlim(2700, 5000)
axes.legend()
fig.savefig(f"figs/testing/mass_dist_{signal}_{(int(pct * 1000) % 100):02d}.png")

# Plot
nbins = 30
fig, axes = plt.subplots(figsize=(8,6))
axes.hist([loss_sample_total], nbins, range=(0, 1.5), density=1, histtype='step', label=['No selection'], stacked=True, alpha=0.6)
axes.hist([loss_sample_total[loss_sample_total > threshold[50 - 1]]], nbins, range=(0, 1.5), density=1, histtype='step', label=['50%'], stacked=True, alpha=0.8)
axes.hist([loss_sample_total[loss_sample_total > threshold[99 - 1]]], nbins, range=(0, 1.5), density=1, histtype='step', label=['99%'], stacked=True, alpha=1)
axes.set_xlabel(r"$m_{jet_1•jet_2}$ [GeV]")
axes.set_ylabel("Events")
axes.set_xlim(0, 1.5)
axes.legend()
fig.savefig(f"figs/testing/loss_{signal}_{(int(pct * 1000) % 100):02d}.png")

############################################ Jensen Shannon Distribution  ##############################################

# Reference uncut histogram
hist_ref, bins = np.histogram(mjj_sample[:100000], bins=30, range=scope)

# Loop over percentiles
jsd = []
for th in threshold:
    hist_cut, _ = np.histogram(mjj_sample[:100000][loss_sample_total > th], bins=bins, range=scope)
    jsd.append(jensenshannon(hist_cut, hist_ref))

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
loss_sig_avg = []

for i in range(len(bins) - 1):
    loss_sample_bin = loss_sample_total[(mjj_sample[:100000] >= bins[i]) & (mjj_sample[:100000] < bins[i + 1])]
    loss_sig_bin = loss_sig_total[(mjj_sig >= bins[i]) & (mjj_sig < bins[i + 1])]

    loss_sample_bin_avg = np.mean(loss_sample_bin)
    loss_sig_bin_avg = np.mean(loss_sig_bin)

    loss_sample_avg.append(loss_sample_bin_avg)
    loss_sig_avg.append(loss_sig_bin_avg)

# Plot Avg Loss v. Mass
fig, axes = plt.subplots(figsize=(8,6))
axes.plot(bins[:-1] + np.diff(bins) / 2, loss_sample_avg, label='Background')
axes.plot(bins[:-1] + np.diff(bins) / 2, loss_sig_avg, label='Signal')
axes.set_xlim([2700, 5000])
axes.set_xlabel(r"$m_{jet_1•jet_2}$")
axes.set_ylabel('Reconstruction Error')
axes.set_title('Avg Error v. Mass Distribution')
axes.legend()
fig.savefig(f"figs/testing/AvgLossMass_{signal}_{(int(pct * 1000) % 100):02d}.png")


#########

# nbins = 50
# fig, axes = plt.subplots(figsize=(8,6))
# axes.hist([mjj_bkg], nbins, histtype='step', weights=weights_bkg.cpu().numpy(), label=['Background'], stacked=True, alpha=1)
# axes.set_xlabel(r"$m_{jet_1•jet_2}$")
# axes.set_ylabel("Events")
# axes.legend()
# fig.savefig(f"figs/testing/normalised_mass_dist_{scale}_{mid_dim}_{latent_dim}.png")