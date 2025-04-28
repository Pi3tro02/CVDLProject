import matplotlib.pyplot as plt
import torch

def plot_batch(loader, device="cpu"):
    batch = next(iter(loader))
    images = batch["image"].to(device)
    labels = batch["label"].to(device)

    # Prendiamo il primo esempio nel batch
    image = images[0, 0]  # [B, C, D, H, W] -> prendiamo batch 0, canale 0
    label = labels[0, 0]

    # Scegliamo uno slice centrale lungo l'asse depth (D)
    slice_idx = image.shape[0] // 2

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Image slice")
    plt.imshow(image[slice_idx].cpu().numpy(), cmap="gray")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Label slice")
    plt.imshow(label[slice_idx].cpu().numpy(), cmap="gray")
    plt.axis('off')

    plt.show()
