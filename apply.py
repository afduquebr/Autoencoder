#######################################################################################################
"""
Created on Jul 05 2024

@author: Andres Felipe DUQUE BRAN
"""
#######################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import roc_curve, auc
from scipy.spatial.distance import jensenshannon
import pyBumpHunter as bh

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
    sample_bkg = bkg.sample(frac=1)
    sample_sig = globals()[signal].sample(frac=1)
    sample = pd.concat([sample_bkg, sample_sig[:int(pct * len(bkg))]])
    sample['labels'] = pd.Series([0]*len(sample_bkg) + [1]*len(sample_sig[:int(pct * len(bkg))]))
    sample = sample.sample(frac=1)
else:
    signal = "sig1"
    pct = 0
    sample_bkg = bkg.sample(frac=1)
    sample_sig = sig1.sample(frac=1)
    sample = sample_bkg
    sample['labels'] = pd.Series([0]*len(sample))

mjj_sample = sample[mass].values

# Print original S/B ratio
labels = sample['labels']
sbr = labels[labels==1].size / labels[labels==0].size
print(f'Original S/B = {100 * sbr:.2f}%')


#######################################################################################################
######################################## Data Preprocessing ###########################################

# Concatenate all datasets for the current column to find the global min and max
all_data = sample[selection]

for col in smooth_cols:
    first_positive = all_data[col][all_data[col] > 0].min()
    all_data[col] = np.where(all_data[col] <= 0, first_positive, all_data[col])

all_data[smooth_cols] = all_data[smooth_cols].apply(lambda x: np.log(x))

# Create a Scaler object with adjusted parameters for each column
scaler = StandardScaler()
sample = pd.DataFrame(scaler.fit_transform(all_data), columns=selection)

data = torch.from_numpy(sample[:100000].values).float().to(device)
pt_mjj = torch.from_numpy(mjj_sample[:100000]).float().to(device)

#######################################################################################################
############################################# Analysis ################################################

# Latent space dimension (embedding)
input_dim = selection.size

# Load Model
model = AutoEncoder(input_dim = input_dim).to(device)
model.load_state_dict(torch.load(f"models/model_parameters_{signal}_{(int(pct * 1000) % 100):02d}.pth", map_location=device))
model.eval()

# Predictions
with torch.no_grad(): # no need to compute gradients here
    prediction = model(data)

# MSE 
loss_sample = pd.DataFrame(loss(data, prediction).numpy(), columns=selection).mean(axis=1)

# Do the selection at Nth percentile
percentile = 85
cut = np.percentile(loss_sample, percentile)
mjj_sample_cut = mjj_sample[loss_sample > cut]
print(f'    post cut stat : {mjj_sample_cut.size}')
print(f'    selec threshold : {cut:.4f}')

# Compute signal efficiency (based on truth) if possible
labels_cut = labels[loss_sample > cut]
sig_eff =  mjj_sample_cut[labels_cut == 1].size / mjj_sample[labels == 1].size
print(f'    sig_eff : {sig_eff}')
sbr = mjj_sample_cut[labels_cut == 1].size / mjj_sample_cut[labels_cut == 0].size
print(f'    new S/B : {100 * sbr:.2f}%')


#######################################################################################################
############################################# Analysis ################################################

BH = bh.BumpHunter1D(
    rang=scope,
    width_min = 3,
    width_max = 8,
    width_step = 1,
    scan_step = 1,
    npe = 40000,
    bins = 40,
    nworker = 4,
    # use_sideband = True,
    # sideband_width = [6, 4],
    seed = 500
)

# Do the BH scan
BH.bump_scan(mjj_sample_cut, mjj_sample)
print(BH.bump_info(mjj_sample_cut))

# Plot results
BH.plot_tomography(mjj_sample, filename="figs/BumpHunter/tomography.png")
BH.plot_bump(mjj_sample_cut, mjj_sample, filename="figs/BumpHunter/bump.png")
BH.plot_stat(show_Pval=True, filename="figs/BumpHunter/BHstat.png")



