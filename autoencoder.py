#######################################################################################################
"""
Created on Mar 25 2024

@author: Andres Felipe DUQUE BRAN
"""
#######################################################################################################

import torch
import torch.nn as nn
from Disco import distance_corr

####################################### GPU or CPU running ###########################################

# Select the device for computation (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#######################################################################################################
######################################### AutoEncoder Class ###########################################

# Define the AutoEncoder class
class AutoEncoder(nn.Module):
    def __init__(self, input_dim = 42, mid_dim = 21, latent_dim = 14):
        super(AutoEncoder, self).__init__()
        # Define the encoder part of the autoencoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, mid_dim), # Linear layer from input to middle dimension
            nn.ReLU(),                     # ReLU activation function
            nn.Linear(mid_dim, latent_dim),# Linear layer from middle to latent dimension
            nn.ReLU()                      # ReLU activation function
        )
        
        # Define the decoder part of the autoencoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, mid_dim),# Linear layer from latent to middle dimension
            nn.ReLU(),                     # ReLU activation function
            nn.Linear(mid_dim, input_dim), # Linear layer from middle to input dimension
            nn.PReLU()                     # PReLU activation function
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
        # Forward pass through the encoder
        x = self.encoder(x)
        return x
            
    def decode(self, x):
        # Forward pass through the decoder
        x = self.decoder(x)
        return x

    def forward(self, x):
        # Forward pass through the entire autoencoder
        x = self.encode(x)
        x = self.decode(x)
        # Re-apply the constraint on the weights
        self.constrain_weights()
        return x

#######################################################################################################
########################################### Weighted Loss #############################################

# Define a function for weighted mean squared error loss
def WeightedMSELoss(output, target, weight):
    # Calculate the weighted MSE loss
    loss_MSE = torch.mean(weight.unsqueeze(1) * (output - target)**2)
    return loss_MSE

#######################################################################################################
########################################## Model Training #############################################

# Define the training function
def train(model, data_loader, loss_function, opt, epoch, alpha=0):
    model.train() # Set the model to training mode
    for i, (features, weights, mass) in enumerate(data_loader):
        features = features.to(device) # Move features to the device
        mass = mass.to(device)         # Move mass to the device
        prediction = model(features)   # Get the model's predictions
        error = torch.mean(loss(features, prediction), dim=1) # Calculate reconstruction error
        disco = distance_corr(mass, error, torch.ones_like(mass)) # Calculate distance correlation
        train_loss = loss_function(prediction, features, weights) + alpha * disco # Total loss
        opt.zero_grad() # Zero the gradients
        train_loss.backward() # Backpropagate the error
        opt.step() # Update the model parameters

        # Print statistics every 10 batches
        if i % 10 == 9:    
            print('[Epoch : %d, iteration: %5d]' % (epoch + 1, (i + 1) + epoch * len(data_loader.dataset)))
            print('Training loss: %.3f' % (train_loss.item()))
    return train_loss.item()

#######################################################################################################
##################################### Model Testing and Loss ##########################################

# Define the testing function
def test(model, data_loader, loss_function, epoch):
    model.eval() # Set the model to evaluation mode
    for i, (features, _) in enumerate(data_loader):     
        features = features.to(device) # Move features to the device
        prediction = model(features)   # Get the model's predictions
        test_loss = loss_function(prediction, features) # Calculate the test loss
        # Print statistics every 10 batches
        if i % 10 == 9:    
            print('[Epoch : %d, iteration: %5d]' % (epoch + 1, (i + 1) + epoch * len(data_loader.dataset)))
            print('Testing loss: %.3f' % (test_loss.item()))
    return test_loss.item()

# Define Reconstruction Error function
def loss(output, target):
    # Calculate the squared difference between output and target
    return torch.pow(output - target, 2)
