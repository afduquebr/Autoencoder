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
        # Parse command line arguments and call main function
        self.path, self.signal, self.pct = parse_args()
        main()
        # Load and preprocess data
        self.load_data()
        self.preprocess_data()
        self.reweight_data()

    def load_data(self):
        # Define the path to the data and various selection criteria
        self.path = "../LHCO-Dataset/data"
        # Define relevant features for training
        self.selection = pd.read_csv("selection/dijet-selection.csv", header=None).values[:, 0]
        # Define features for additional rescaling with log
        self.smooth_cols = pd.read_csv("selection/scale-selection.csv", header=None).values[:, 0]
        self.mass = 'mj1j2'
        # Define relevant mass range
        self.scope = [2700, 5000]
        # Define masses and taus for additional cut
        self.masses = ["mass_1", "mass_2"]
        self.tau = ["tau21_1", "tau21_2", "tau32_1", "tau32_2"]

        # Load background and signal datasets
        self.bkg = pd.read_hdf(f"{self.path}/RnD_scalars_bkg.h5")
        self.sig = pd.read_hdf(f"{self.path}/RnD_scalars_sig.h5")
        self.bbox1 = pd.read_hdf(f"{self.path}/BBOX1_scalars_sig.h5")
        self.bbox2 = pd.read_hdf(f"{self.path}/BBOX2_scalars_bkg.h5")

        # Handle missing or infinite values
        for df in [self.bkg, self.sig, self.bbox1, self.bbox2]:
            # Replace NaN, -inf, and inf values with 0
            df.replace([np.nan, -np.inf, np.inf], 0, inplace=True)
            # Filter rows based on mass and tau criteria
            df = df[(df[self.mass] > self.scope[0]) & (df[self.mass] < self.scope[1])].reset_index(drop=True)
            df = df[(df[self.masses] >= 5.0).all(axis=1)].reset_index(drop=True)
            df = df[(df[self.tau] >= 0).all(axis=1) & (df[self.tau] <= 1).all(axis=1)].reset_index(drop=True)

    def mix_signal(self):
        # If a signal is specified, mix signal and background data
        if self.signal:
            self.sample_bkg = self.bkg.sample(frac=1, ignore_index=True)
            self.sample_sig = getattr(self, self.signal).sample(frac=1, ignore_index=True)
            sample = pd.concat([self.sample_bkg, self.sample_sig[:int(self.pct * len(self.bkg))]], ignore_index=True)
            sample['labels'] = pd.Series([0]*len(self.sample_bkg) + [1]*len(sample[len(self.sample_bkg):]))
            self.sample = sample.sample(frac=1, ignore_index=True)
        else:
            # Default signal and sample values if no signal is specified
            self.signal = "sig1"
            self.pct = 0
            self.sample_sig = self.sig.sample(frac=1, ignore_index=True)
            self.sample = self.bkg.sample(frac=1, ignore_index=True)
            self.sample['labels'] = pd.Series([0]*len(self.sample))

        # Extract mass values for the mixed sample, background, and signal
        self.mjj_sample = self.sample[self.mass].values
        self.mjj_bkg = self.sample_bkg[self.mass].values
        self.mjj_sig = self.sample_sig[self.mass].values

        # Extract labels
        self.labels = self.sample['labels']

    def reweight_data(self):
        # Compute histogram for reweighting
        Hc, Hb = np.histogram(self.mjj_sample, bins=500)
        weights = np.array(Hc, dtype=float)
        # Inverse frequency for weights, set zero-weight bins to 1
        weights[weights > 0.0] = 1.0 / weights[weights > 0.0]
        weights[weights == 0.0] = 1.0
        weights = np.append(weights, weights[-1])
        # Scale weights for normalization
        weights *= 1000.0
        self.weights = weights[np.searchsorted(Hb, self.mjj_sample)]

    def preprocess_data(self):
        # Mix the signal and background samples
        self.mix_signal()
        
        # Concatenate all datasets for the current column to find the global min and max
        all_data = pd.concat([self.sample[self.selection], self.sample_bkg[self.selection], self.sample_sig[self.selection]])

        for col in self.smooth_cols:
            # Replace non-positive values with the minimum positive value in the column
            first_positive = all_data[col][all_data[col] > 0].min()
            all_data.loc[all_data[col] <= 0, col] = first_positive

        # Apply log transformation to smooth columns
        all_data.loc[:, self.smooth_cols] = all_data.loc[:, self.smooth_cols].apply(np.log)

        # Create a Scaler object with adjusted parameters for each column
        scaler = StandardScaler()
        data_scaled = pd.DataFrame(scaler.fit_transform(all_data), columns=self.selection)

        # Apply scaling to each dataset per column
        self.sample_scaled = data_scaled.iloc[:len(self.sample)].reset_index(drop=True)
        self.bkg_scaled = data_scaled.iloc[len(self.sample):-len(self.sample_sig)].reset_index(drop=True)
        self.sig_scaled = data_scaled.iloc[-len(self.sample_sig):].reset_index(drop=True)

    def get_scaled_data(self):
        # Return scaled data for sample, background, and signal
        return self.sample_scaled, self.bkg_scaled, self.sig_scaled
    
    def get_mass(self):
        # Return mass values for sample, background, and signal
        return self.mjj_sample, self.mjj_bkg, self.mjj_sig
    
    def get_weights(self):
        # Return computed weights for the samples
        return self.weights
    
    def get_labels(self):
        # Return labels for the samples
        return self.labels
