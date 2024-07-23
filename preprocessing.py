#######################################################################################################
"""
Created on Jul 23 2024

@author: Andres Felipe DUQUE BRAN
"""
#######################################################################################################

# preprocessing.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch

class Preprocessor:
    def __init__(self, path, signal, pct, selection_file, smooth_cols_file):
        self.path = path
        self.signal = signal
        self.pct = pct
        self.selection = pd.read_csv(selection_file, header=None).values[:, 0]
        self.smooth_cols = pd.read_csv(smooth_cols_file, header=None).values[:, 0]
        self.mass = 'mj1j2'
        self.scope = [2700, 5000]
        self.masses = ["mass_1", "mass_2"]
        self.tau = ["tau21_1", "tau21_2", "tau32_1", "tau32_2"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)
        self.load_data()
        self.preprocess_data()

    def load_data(self):
        if self.path == "local":
            self.path = "../GAN-AE/clustering-lhco/data"
        elif self.path == "server": 
            self.path = "/AtlasDisk/user/duquebran/clustering-lhco/data"

        self.bkg = pd.read_hdf(f"{self.path}/RnD_2j_scalars_bkg.h5")
        self.sig1 = pd.read_hdf(f"{self.path}/RnD_2j_scalars_sig.h5")
        self.sig2 = pd.read_hdf(f"{self.path}/RnD2_2j_scalars_sig.h5")
        self.bbox = pd.read_hdf(f"{self.path}/BBOX1_2j_scalars_sig.h5")

        # Handle missing or infinite values
        for df in [self.bkg, self.sig1, self.sig2, self.bbox]:
            df.replace([np.nan, -np.inf, np.inf], 0, inplace=True)

        # Apply mass scope filtering
        for df in [self.bkg, self.sig1, self.sig2, self.bbox]:
            df = df[(df[self.mass] > self.scope[0]) & (df[self.mass] < self.scope[1])].reset_index(drop=True)

        # Apply additional filtering on mass and tau values
        for df in [self.bkg, self.sig1, self.sig2, self.bbox]:
            df = df[(df[self.masses] >= 5.0).all(axis=1)].reset_index(drop=True)
            df = df[(df[self.tau] >= 0).all(axis=1) & (df[self.tau] <= 1).all(axis=1)].reset_index(drop=True)

        self.bkg, self.sig1, self.sig2, self.bbox = [self.bkg, self.sig1, self.sig2, self.bbox]

    def mix_signal(self):
        if self.signal:
            self.sample_bkg = self.bkg.sample(frac=1).reset_index(drop=True)
            self.sample_sig = getattr(self, self.signal).sample(frac=1).reset_index(drop=True)
            self.sample = pd.concat([self.sample_bkg, self.sample_sig[:int(self.pct * len(self.bkg))]]).sample(frac=1).reset_index(drop=True)
        else:
            self.signal = "sig1"
            self.pct = 0
            self.sample_sig = self.sig1.sample(frac=1).reset_index(drop=True)
            self.sample = self.bkg.sample(frac=1).reset_index(drop=True)

    def preprocess_data(self):
        self.mix_signal()
        
        # Concatenate all datasets for the current column to find the global min and max
        all_data = pd.concat([self.sample[self.selection], self.sample_bkg[self.selection], self.sample_sig[self.selection]])

        for col in self.smooth_cols:
            first_positive = all_data[col][all_data[col] > 0].min()
            all_data.loc[all_data[col] <= 0, col] = first_positive

        all_data.loc[:, self.smooth_cols] = all_data.loc[:, self.smooth_cols].apply(np.log)

        # Create a Scaler object with adjusted parameters for each column
        scaler = StandardScaler()
        data_scaled = pd.DataFrame(scaler.fit_transform(all_data), columns=self.selection)

        # Apply scaling to each dataset per column
        self.sample_scaled = data_scaled.iloc[:len(self.sample)]
        self.bkg_scaled = data_scaled.iloc[len(self.sample):-len(self.sample_sig)]
        self.sig_scaled = data_scaled.iloc[-len(self.sample_sig):]

    def get_scaled_data(self):
        return self.sample_scaled, self.bkg_scaled, self.sig_scaled
