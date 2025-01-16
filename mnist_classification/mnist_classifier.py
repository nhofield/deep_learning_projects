import torch
import torchvision

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

