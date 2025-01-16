import torch
import torchvision
import torch.nn as nn

# fetch mnist dataset
mnist_dataset = torchvision.datasets.MNIST(
    root = "./data", 
    train = True, 
    transform = torchvision.transforms.ToTensor(), 
    download = True 
    )

# load the data
train_loader = torch.utils.data.DataLoader(
    dataset = mnist_dataset,
    batch_size = 64,
    shuffle = True
    )

class SimpleNN(torch.nn.Module):

    def __init__(self):

        super(SimpleNN, self).__init__()
        self.initial_layer  = nn.Linear(28*28, 128)
        self.hidden_layer_1 = nn.Linear(128, 64)
        self.hidden_layer_2 = nn.Linear(64, 10)
    
    def forward(self):

        x = x.view(-1, 28 * 28)

        # apply layers and activations
        x = nn.ReLU()(self.initial_layer(x))
        x = nn.ReLU()(self.hidden_layer_1(x))
        x = self.hidden_layer_2(x)  # No activation on final layer


