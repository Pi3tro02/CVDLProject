import matplotlib.pyplot as plt
import torch

## Function that takes as input a Pytorch DataLoader and the device, and it visualizes an example of image and mask
def plot_batch(loader, device="cpu"):
    # Selects the first batch from the loader and moves the image and the label to the device
    batch = next(iter(loader))
    images = batch["image"].to(device)
    labels = batch["label"].to(device)

    # Takes the first example
    image = images[0, 0]  # [B, C, D, H, W]
    label = labels[0, 0]

    # Selects the central slice for a 2D section
    slice_idx = image.shape[0] // 2

    # Creates a figure where: on the left there is the 2D image and on the right there is the label (tumor mask)
    plt.figure(figsize=(12, 6))
    # Image
    plt.subplot(1, 2, 1)
    plt.title("Image slice")
    plt.imshow(image[slice_idx].cpu().numpy(), cmap="gray")
    plt.axis('off')

    # Label
    plt.subplot(1, 2, 2)
    plt.title("Label slice")
    plt.imshow(label[slice_idx].cpu().numpy(), cmap="gray")
    plt.axis('off')

    plt.show()
