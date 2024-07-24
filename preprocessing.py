#######################################################################################################
"""
Created on Jul 23 2024

@author: Andres Felipe DUQUE BRAN
"""
#######################################################################################################

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from main import main, parse_args


class Preprocessor:
    def __init__(self):
        self.path, self.signal, self.pct = parse_args()
        main()
        self.load_data()
        self.preprocess_data()
        self.reweight_data()


    def load_data(self):
        if self.path == "local":
            self.path = "../GAN-AE/clustering-lhco/data"
        elif self.path == "server": 
            self.path = "/AtlasDisk/user/duquebran/clustering-lhco/data"

        
        self.selection = pd.read_csv("dijet-selection.csv", header=None).values[:, 0]
        self.smooth_cols = pd.read_csv("scale-selection.csv", header=None).values[:, 0]
        self.mass = 'mj1j2'
        self.scope = [2700, 5000]
        self.masses = ["mass_1", "mass_2"]
        self.tau = ["tau21_1", "tau21_2", "tau32_1", "tau32_2"]

        self.bkg = pd.read_hdf(f"{self.path}/RnD_2j_scalars_bkg.h5")
        self.sig1 = pd.read_hdf(f"{self.path}/RnD_2j_scalars_sig.h5")
        self.sig2 = pd.read_hdf(f"{self.path}/RnD2_2j_scalars_sig.h5")
        self.bbox = pd.read_hdf(f"{self.path}/BBOX1_2j_scalars_sig.h5")

        # Handle missing or infinite values
        for df in [self.bkg, self.sig1, self.sig2, self.bbox]:
            df.replace([np.nan, -np.inf, np.inf], 0, inplace=True)
            df = df[(df[self.mass] > self.scope[0]) & (df[self.mass] < self.scope[1])].reset_index(drop=True)
            df = df[(df[self.masses] >= 5.0).all(axis=1)].reset_index(drop=True)
            df = df[(df[self.tau] >= 0).all(axis=1) & (df[self.tau] <= 1).all(axis=1)].reset_index(drop=True)

    def mix_signal(self):
        if self.signal:
            self.sample_bkg = self.bkg.sample(frac=1, ignore_index=True)
            self.sample_sig = getattr(self, self.signal).sample(frac=1, ignore_index=True)
            sample = pd.concat([self.sample_bkg, self.sample_sig[:int(self.pct * len(self.bkg))]], ignore_index=True)
            sample['labels'] = pd.Series([0]*len(self.sample_bkg) + [1]*len(sample[len(self.sample_bkg):]))
            self.sample = sample.sample(frac=1, ignore_index=True)
        else:
            self.signal = "sig1"
            self.pct = 0
            self.sample_sig = self.sig1.sample(frac=1, ignore_index=True)
            self.sample = self.bkg.sample(frac=1, ignore_index=True)
            self.sample['labels'] = pd.Series([0]*len(self.sample))

        self.mjj_sample = self.sample[self.mass].values
        self.mjj_bkg = self.sample_bkg[self.mass].values
        self.mjj_sig = self.sample_sig[self.mass].values

        self.labels = self.sample['labels']

    def reweight_data(self):
        Hc, Hb = np.histogram(self.mjj_sample, bins=500)
        weights = np.array(Hc, dtype=float)
        weights[weights > 0.0] = 1.0 / weights[weights > 0.0]
        weights[weights == 0.0] = 1.0
        weights = np.append(weights, weights[-1])
        weights *= 1000.0
        self.weights = weights[np.searchsorted(Hb, self.mjj_sample)]


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
        self.sample_scaled = data_scaled.iloc[:len(self.sample)].reset_index(drop=True)
        self.bkg_scaled = data_scaled.iloc[len(self.sample):-len(self.sample_sig)].reset_index(drop=True)
        self.sig_scaled = data_scaled.iloc[-len(self.sample_sig):].reset_index(drop=True)

    def get_scaled_data(self):
        return self.sample_scaled, self.bkg_scaled, self.sig_scaled
    
    def get_mass(self):
        return self.mjj_sample, self.mjj_bkg, self.mjj_sig
    
    def get_weights(self):
        return self.weights
    
    def get_labels(self):
        return self.labels
