#######################################################################################################
"""
Created on Jun 10 2024

@author: Andres Felipe DUQUE BRAN
"""
#######################################################################################################

import os
import matplotlib.pyplot as plt
import torch

from autoencoder import AutoEncoder
from preprocessing import Preprocessor
from main import parse_args

####################################### GPU or CPU running ###########################################

# Check if CUDA is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#######################################################################################################
####################################### Data Preprocessing ###########################################

# Parse command line arguments to get signal type and percentage
signal, pct = parse_args()
# Initialize the Preprocessor class
preprocessing = Preprocessor()
# Get scaled data from the preprocessor
sample_scaled, _, sig_scaled = preprocessing.get_scaled_data()

#######################################################################################################
######################################### Data Rescaling ##############################################

# Convert the scaled data to PyTorch tensors and move to the appropriate device
test_sample = torch.from_numpy(sample_scaled[:100000].values).float().to(device)
test_sig = torch.from_numpy(sig_scaled[:100000].values).float().to(device)

#######################################################################################################
########################################## Testing Analysis ############################################

# Get the number of input dimensions for the AutoEncoder
input_dim = preprocessing.selection.size

# Load the trained AutoEncoder model
model = AutoEncoder(input_dim=input_dim).to(device)
model.load_state_dict(torch.load(f"models/{signal}/parameters_{(int(pct * 10) % 100):02d}.pth", map_location=device))
model.eval()  # Set the model to evaluation mode

# Make predictions without calculating gradients
with torch.no_grad(): 
    predict_sample = model(test_sample)
    predict_sig = model(test_sig)

#######################################################################################################
############################################# Histograms ##############################################

# Create directory for storing histogram plots
directory = f"figs/histograms/{signal}/{(int(pct * 10) % 100):02d}"

if not os.path.exists(directory):
    os.makedirs(directory)

# Set the number of bins for the histograms
nbins = 20

# Loop over each selected feature for creating histograms
for i, column in enumerate(preprocessing.selection):
    fig, axes = plt.subplots(figsize=(8, 6))
    
    # Plot histogram for original and predicted background data
    axes.hist([test_sample.cpu().numpy()[:, i]], nbins, density=0, histtype='step', label=['Background'], stacked=True, alpha=1)
    axes.hist([predict_sample.cpu().numpy()[:, i]], nbins, density=0, histtype='step', label=['BKG prediction'], stacked=True, alpha=0.3)
    
    # Plot histogram for original and predicted signal data
    axes.hist([test_sig.cpu().numpy()[:, i]], nbins, density=0, histtype='step', label=['Signal'], stacked=True, alpha=1)
    axes.hist([predict_sig.cpu().numpy()[:, i]], nbins, density=0, histtype='step', label=['Signal prediction'], stacked=True, alpha=0.3)
    
    # Set labels and title for the plot
    axes.set_xlabel(f"{column}")
    axes.set_ylabel("Events")
    axes.set_title(f"Prediction of {column}")
    axes.set_yscale("log")
    
    # Add legend to the plot
    fig.legend()
    
    # Save the plot as an image file
    fig.savefig(f"{directory}/hist_{column}.png")
    
    # Close the plot to free up memory
    plt.close()