#######################################################################################################
"""
Created on Mar 25 2024

@author: Andres Felipe DUQUE BRAN
"""
#######################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision

from Disco import distance_corr

####################################### GPU or CPU running ###########################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#######################################################################################################
######################################### AutoEncoder Class ###########################################

# Creating a PyTorch class
# 42 ==> 84 ==> 14 ==> 84 ==> 42
class AutoEncoder(nn.Module):
    def __init__(self, input_dim = 42, mid_dim = 21, latent_dim = 14):
        super(AutoEncoder, self).__init__()
        # Building an linear encoder with Linear
        # layer followed by Relu activation function
        # 42 ==> 14
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, latent_dim),
            nn.ReLU()
        )
        
        # Building an linear decoder with Linear
        # layer followed by Relu activation function
        # 14 ==> 42
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, input_dim),
            # nn.ReLU() # MinMax Scaling 
            nn.PReLU() # Standard Scaling
        )

        # Initialize decoder weights with encoder weights
        self.constrain_weights() 
        
    def constrain_weights(self):
        # Iterate over encoder and decoder layers
        for encoder_layer, decoder_layer in zip(self.encoder, reversed(self.decoder)):
            # Check if the layer is a linear layer
            if isinstance(encoder_layer, nn.Linear) and isinstance(decoder_layer, nn.Linear):
                # Assign encoder's weights to decoder's weights
                decoder_layer.weight.data = encoder_layer.weight.data.clone().t()


    def encode(self, x):
        x = self.encoder(x)
        return x
            
    def decode(self, x):
        x = self.decoder(x)
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x) 
        self.constrain_weights()
        return x

#######################################################################################################
########################################### Weighted Loss #############################################

def WeightedMSELoss(output, target, weight):
    loss_MSE = torch.mean(weight.unsqueeze(1) * (output - target)**2)
    return loss_MSE

#######################################################################################################
########################################## Model Training #############################################

def train(model, data_loader, loss_function, opt, epoch, alpha=0):
    model.train()
    for i, (features, weights, mass) in enumerate(data_loader): 
        features = features.to(device)
        mass = mass.to(device)
        prediction = model(features)
        error = torch.mean(loss(features, prediction), dim=1)
        disco = distance_corr(mass, error, torch.ones_like(mass))
        train_loss = loss_function(prediction, features, weights) + alpha * disco
        opt.zero_grad()
        train_loss.backward()
        opt.step()

        # print statistics
        if i % 100 == 99:    
            print('[Epoch : %d, iteration: %5d]'% (epoch + 1, (i + 1) + epoch * len(data_loader.dataset)))
            print('Training loss: %.3f'% (train_loss.item()))
    return train_loss.item()

#######################################################################################################
##################################### Model Testing and Loss ##########################################

def test(model, data_loader, loss_function, epoch):
    model.eval()
    for i, (features, _) in enumerate(data_loader):     
        features = features.to(device) 
        prediction = model(features)
        test_loss = loss_function(prediction, features)
        # print statistics
        if i % 100 == 99:    
            print('[Epoch : %d, iteration: %5d]'% (epoch + 1, (i + 1) + epoch * len(data_loader.dataset)))
            print('Testing loss: %.3f'% (test_loss.item()))
    return test_loss.item()

# Define Reconstruction Error function
def loss(output, target):
    return torch.pow(output - target, 2)
