import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn

from utils.visualization import show_6_images

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
    
        # Define activation
        self.relu = nn.ReLU()

    def forward(self, x):

        x = x.view(-1, 28 * 28)

        # apply layers and activations
        x = self.relu(self.initial_layer(x))
        x = self.relu(self.hidden_layer_1(x))
        x = self.hidden_layer_2(x)  

        return x

model = SimpleNN()

# Check if a GPU is available, otherwise fall back to CPU
device = torch.device("cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

show_6_images(model, device, train_loader)

num_epochs = 5
loss_list = []

for epoch in range(num_epochs):  # Loop over epochs
    for images, labels in train_loader:  # Loop over batches
        # Move data to the device
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass: Compute predictions
        outputs = model(images)
        
        # Compute the loss
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())  # Use .item() to extract the scalar value
        
        # Backward pass: Compute gradients
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Clear gradients
        optimizer.zero_grad()


test_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=False,  # Load the test dataset
    transform=torchvision.transforms.ToTensor(),
    download=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=64,
    shuffle=False  # No need to shuffle test data
)

correct = 0
total = 0

# Set the model to evaluation mode
model.eval()

# Disable gradient computation
with torch.no_grad():
    for images, labels in test_loader:
        # Move data to the appropriate device
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        
        # Get predictions (class with highest score)
        _, predicted = torch.max(outputs, 1)
        
        # Update total and correct counts
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Calculate accuracy
accuracy = 100 * correct / total
print(f"Accuracy on the test dataset: {accuracy:.2f}%")

# Save the model's state dictionary
torch.save(model.state_dict(), "models/mnist_model.pth")
