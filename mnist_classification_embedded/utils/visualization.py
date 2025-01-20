import matplotlib.pyplot as plt
import torch
import torchvision

def show_6_images(model, device, train_loader):

    images, labels = next(iter(train_loader))

    # Move the data to the same device as the model
    images, labels = images.to(device), labels.to(device)

    # Pass the images through the model
    output = model(images)

    # Print the shapes
    print(f"Input batch shape: {images.shape}")  # Should be [batch_size, 1, 28, 28]
    print(f"Output batch shape: {output.shape}")  # Should be [batch_size, 10]

    # Plot the first 6 images in the batch
    plt.figure(figsize=(10, 5))
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i].squeeze().cpu().numpy(), cmap="gray")  # Remove channel, move to CPU
        plt.title(f"Label: {labels[i].item()}")
        plt.axis("off")
    plt.show()
