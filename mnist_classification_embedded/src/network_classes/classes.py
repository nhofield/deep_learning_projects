import torch
import torch.nn as nn

class SimpleNN(torch.nn.Module):

    def __init__(self, picture_dim, hidden_layer_neurons):

        super(SimpleNN, self).__init__()
        self.initial_layer  = nn.Linear(picture_dim**2, hidden_layer_neurons)
        self.hidden_layer = nn.Linear(hidden_layer_neurons, 10)              # Output layer with 10 neurons
        self.relu = nn.ReLU()                               # Activation function
    
    def forward(self, x):

        x = x.view(x.size(0), -1)
        x = self.relu( self.initial_layer(x) )
        x = self.hidden_layer(x)
    
        return x
