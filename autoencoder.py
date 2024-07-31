#######################################################################################################
"""
Created on Mar 25 2024

@author: Andres Felipe DUQUE BRAN
"""
#######################################################################################################

import torch
import torch.nn as nn

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

#########################################################################################################
########################################### Distance Correlation Function ###############################
"""
Taken from https://github.com/gkasieczka/DisCo/blob/master/Disco.py
"""

def distance_corr(var_1,var_2,normedweight,power=1):
    """var_1: First variable to decorrelate (eg mass)
    var_2: Second variable to decorrelate (eg classifier output)
    normedweight: Per-example weight. Sum of weights should add up to N (where N is the number of examples)
    power: Exponent used in calculating the distance correlation
    
    va1_1, var_2 and normedweight should all be 1D torch tensors with the same number of entries
    
    Usage: Add to your loss function. total_loss = BCE_loss + lambda * distance_corr
    """
    
    
    xx = var_1.view(-1, 1).repeat(1, len(var_1)).view(len(var_1),len(var_1))
    yy = var_1.repeat(len(var_1),1).view(len(var_1),len(var_1))
    amat = (xx-yy).abs()

    xx = var_2.view(-1, 1).repeat(1, len(var_2)).view(len(var_2),len(var_2))
    yy = var_2.repeat(len(var_2),1).view(len(var_2),len(var_2))
    bmat = (xx-yy).abs()

    amatavg = torch.mean(amat*normedweight,dim=1)
    Amat=amat-amatavg.repeat(len(var_1),1).view(len(var_1),len(var_1))\
        -amatavg.view(-1, 1).repeat(1, len(var_1)).view(len(var_1),len(var_1))\
        +torch.mean(amatavg*normedweight)

    bmatavg = torch.mean(bmat*normedweight,dim=1)
    Bmat=bmat-bmatavg.repeat(len(var_2),1).view(len(var_2),len(var_2))\
        -bmatavg.view(-1, 1).repeat(1, len(var_2)).view(len(var_2),len(var_2))\
        +torch.mean(bmatavg*normedweight)

    ABavg = torch.mean(Amat*Bmat*normedweight,dim=1)
    AAavg = torch.mean(Amat*Amat*normedweight,dim=1)
    BBavg = torch.mean(Bmat*Bmat*normedweight,dim=1)

    if(power==1):
        dCorr=(torch.mean(ABavg*normedweight))/torch.sqrt((torch.mean(AAavg*normedweight)*torch.mean(BBavg*normedweight)))
    elif(power==2):
        dCorr=(torch.mean(ABavg*normedweight))**2/(torch.mean(AAavg*normedweight)*torch.mean(BBavg*normedweight))
    else:
        dCorr=((torch.mean(ABavg*normedweight))/torch.sqrt((torch.mean(AAavg*normedweight)*torch.mean(BBavg*normedweight))))**power
    
    return dCorr