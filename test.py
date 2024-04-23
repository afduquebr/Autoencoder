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

import torch

from autoencoder import AutoEncoder, loss
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

selection = pd.read_csv("dijet-selection.csv", header=None).values[:, 0]

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
weights_bkg = torch.from_numpy(weights_bkg).float().to(device)

#######################################################################################################
########################################## Testing Analysis ############################################

# Latent space dimension (embedding)
input_dim = selection.size

# Load Model
model = AutoEncoder(input_dim = input_dim, mid_dim = mid_dim, latent_dim = latent_dim).to(device)
model.load_state_dict(torch.load(f"models/model_parameters_{scale}_{mid_dim}_{latent_dim}.pth", map_location=device))

# Predictions
with torch.no_grad(): # no need to compute gradients here
    predict_bkg = model(test_bkg)
    predict_sig1 = model(test_sig1)
    predict_sig2 = model(test_sig2)

predict_bkg_df = pd.DataFrame(scaler.inverse_transform(predict_bkg.cpu().numpy()), columns=selection)
predict_sig1_df = pd.DataFrame(scaler.inverse_transform(predict_sig1.cpu().numpy()), columns=selection)
predict_sig2_df = pd.DataFrame(scaler.inverse_transform(predict_sig2.cpu().numpy()), columns=selection)

# Determine Reconstruction Error
loss_bkg = pd.DataFrame()
loss_sig1 = pd.DataFrame()
loss_sig2 = pd.DataFrame()

# MSE per feature
for i, column in enumerate(predict_bkg_df.columns):
    loss_bkg[column] = loss(test_bkg[:, i], predict_bkg[:, i]).numpy()
    loss_sig1[column] = loss(test_sig1[:, i], predict_sig1[:, i]).numpy()
    loss_sig2[column] = loss(test_sig2[:, i], predict_sig2[:, i]).numpy()

# Total MSE
loss_bkg_total = loss_bkg.sum(axis=1) / 42
loss_sig1_total = loss_sig1.sum(axis=1) / 42
loss_sig2_total = loss_sig2.sum(axis=1) / 42

# Plot Total Reconstruction Error
nbins = 20
fig, axes = plt.subplots(figsize=(8,6))
axes.hist([loss_bkg_total], nbins, range=(0, 0.02), density=0, histtype='step', label=['Background'], stacked=True, alpha=1)
axes.hist([loss_sig1_total], nbins, range=(0, 0.02), density=0, histtype='step', label=['Signal 1'], stacked=True, alpha=0.9)
axes.hist([loss_sig2_total], nbins, range=(0, 0.02), density=0, histtype='step', label=['Signal 2'], stacked=True, alpha=0.9)
axes.set_xlabel(r"Reconstruction Error")
axes.set_ylabel("Events")
axes.set_xlim(0, 0.02)
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

# Plot ROC curve
fig, axes = plt.subplots(figsize=(8,6))
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

bkg_tensor = torch.from_numpy(scaler.transform(bkg[selection])).float().to(device)

# Predictions
with torch.no_grad(): # no need to compute gradients here
    all_bkg = model(bkg_tensor)

all_bkg_df = pd.DataFrame(scaler.inverse_transform(all_bkg.cpu().numpy()), columns=selection)

loss_bkg_all = pd.DataFrame()

for i, column in enumerate(selection):
    loss_bkg_all[column] = loss(bkg_tensor[:, i], all_bkg[:, i]).cpu().numpy()

loss_bkg_all_total = loss_bkg_all.sum(axis=1) / 42

# Invariant mass distribution with respect to BKG anomaly score
cumulative_sum = loss_bkg_all_total.sort_values().cumsum()

# Calculate the total sum
total_sum = cumulative_sum.iloc[-1]

# Calculate the threshold for selecting 80 percent of the values
threshold = 0.8 * total_sum

# Select values where the cumulative sum is less than or equal to the threshold
selected_values = loss_bkg_all_total[cumulative_sum <= threshold]

nbins = 30
fig, axes = plt.subplots(figsize=(8,6))
axes.hist([bkg.mj1j2], nbins, range=(2700, 5000), density=1, histtype='step', label=['No selection'], stacked=True, alpha=1)
axes.hist([bkg.mj1j2[cumulative_sum >= 0.85 * total_sum]], nbins, range=(2700, 5000), density=1, histtype='step', label=['85%'], stacked=True, alpha=0.8)
axes.hist([bkg.mj1j2[cumulative_sum >= 0.5 * total_sum]], nbins, range=(2700, 5000), density=1, histtype='step', label=['50%'], stacked=True, alpha=0.6)
axes.set_xlabel(r"$m_{jet_1•jet_2}$ [GeV]")
axes.set_ylabel("Events")
axes.set_xlim(2700, 5000)
axes.legend()
fig.savefig(f"figs/testing/mass_dist_{scale}_{mid_dim}_{latent_dim}.png")

############################################ Normalised Mass Distribution  ##############################################

nbins = 30
fig, axes = plt.subplots(figsize=(8,6))
axes.hist([mjj_bkg], nbins, histtype='step', weights=weights_bkg.cpu().numpy(), label=['Bkg'], stacked=True, alpha=1)
axes.hist([mjj_sig1], nbins, histtype='step', weights=weights_sig1, label=['Signal 1'], stacked=True, alpha=0.8)
axes.hist([mjj_sig2], nbins, histtype='step', weights=weights_sig2, label=['Signal 2'], stacked=True, alpha=0.6)
axes.set_xlabel(r"$m_{jet_1•jet_2}$")
axes.set_ylabel("Events")
axes.legend()
fig.savefig(f"figs/testing/normalised_mass_dist_{scale}_{mid_dim}_{latent_dim}.png")
