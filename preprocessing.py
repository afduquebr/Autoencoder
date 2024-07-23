#######################################################################################################
"""
Created on Jul 23 2024

@author: Andres Felipe DUQUE BRAN
"""
#######################################################################################################

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from autoencoder import AutoEncoder, loss
from main import main, parse_args

class DataPreprocessor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)
        self.path, self.signal, self.pct = parse_args()
        main()
        self.set_path()
        self.load_data()
        self.clean_data()
        self.filter_data()
        self.mix_data()
        self.reweight_data()
        self.preprocess_data()
        
    def set_path(self):
        if self.path == "local":
            self.path = "../GAN-AE/clustering-lhco/data"
        elif self.path == "server": 
            self.path = "/AtlasDisk/user/duquebran/clustering-lhco/data"
            
    def load_data(self):
        self.bkg = pd.read_hdf(f"{self.path}/RnD_2j_scalars_bkg.h5")
        self.sig1 = pd.read_hdf(f"{self.path}/RnD_2j_scalars_sig.h5")
        self.sig2 = pd.read_hdf(f"{self.path}/RnD2_2j_scalars_sig.h5")
        self.bbox = pd.read_hdf(f"{self.path}/BBOX1_2j_scalars_sig.h5")
        self.selection = pd.read_csv("dijet-selection.csv", header=None).values[:, 0]
        self.smooth_cols = pd.read_csv("scale-selection.csv", header=None).values[:, 0]
    
    def clean_data(self):
        for df in [self.bkg, self.sig1, self.sig2, self.bbox]:
            df.replace([np.nan, -np.inf, np.inf], 0, inplace=True)
    
    def filter_data(self):
        mass = 'mj1j2'
        scope = [2700, 5000]
        masses = ["mass_1", "mass_2"]
        tau = ["tau21_1", "tau21_2", "tau32_1", "tau32_2"]
        
        for df in [self.bkg, self.sig1, self.sig2, self.bbox]:
            df = df[(df[mass] > scope[0]) & (df[mass] < scope[1])].reset_index()
            df = df[(df[masses] >= 5.0).all(axis=1)].reset_index()
            df = df[(df[tau] >= 0).all(axis=1) & (df[tau] <= 1).all(axis=1)].reset_index()
    
    def mix_data(self):
        if self.signal is not None:
            sample_bkg = self.bkg.sample(frac=1)
            sample_sig = globals()[self.signal].sample(frac=1)
            self.sample = pd.concat([self.bkg, sample_sig[:int(self.pct * len(self.bkg))]]).sample(frac=1)
        else:
            self.signal = "sig1"
            self.pct = 0
            sample_sig = self.sig1.sample(frac=1)
            self.sample = self.bkg.sample(frac=1)
        
        self.mjj_sample = self.sample['mj1j2'].values
        self.mjj_bkg = sample_bkg['mj1j2'].values
        self.mjj_sig = sample_sig['mj1j2'].values

    def reweight_data(self):
        Hc, Hb = np.histogram(self.mjj_sample, bins=500)
        weights = np.array(Hc, dtype=float)
        weights[weights > 0.0] = 1.0 / weights[weights > 0.0]
        weights[weights == 0.0] = 1.0
        weights = np.append(weights, weights[-1])
        weights *= 1000.0
        self.weights = weights[np.searchsorted(Hb, self.mjj_sample)]

    def preprocess_data(self):
        all_data = pd.concat([self.sample[self.selection], self.bkg[self.selection], self.sig1[self.selection]])

        for col in self.smooth_cols:
            first_positive = all_data[col][all_data[col] > 0].min()
            all_data.loc[all_data[col] <= 0, col] = first_positive
        
        all_data.loc[:, self.smooth_cols] = all_data.loc[:, self.smooth_cols].apply(np.log)
        scaler = StandardScaler()
        data_scaled = pd.DataFrame(scaler.fit_transform(all_data), columns=self.selection)
        
        self.sample_scaled = data_scaled.iloc[:len(self.sample)]
        self.bkg_scaled = data_scaled.iloc[len(self.sample):-len(self.sig1)]
        self.sig_scaled = data_scaled.iloc[-len(self.sig1):]


