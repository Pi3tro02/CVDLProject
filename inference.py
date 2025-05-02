import os
import torch
import matplotlib.pyplot as plt
from monai.inferers import sliding_window_inference

def load_model(model, model_path, device):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def visualize_inference(model, val_loader, device, roi_size=(160, 160, 160), sw_batch_size=4, num_images=3):
    model.eval()
    with torch.no_grad():
        for i, val_data in enumerate(val_loader):
            images = val_data["image"].to(device)
            labels = val_data["label"].to(device)

            val_outputs = sliding_window_inference(images, roi_size, sw_batch_size, model)

            # Plot one slice at z=80
            plt.figure(f"Inference Check {i}", (18, 6))
            plt.subplot(1, 3, 1)
            plt.title(f"Image {i}")
            plt.imshow(images[0, 0, :, :, 80].cpu(), cmap="gray")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.title(f"Ground Truth {i}")
            plt.imshow(labels[0, 0, :, :, 80].cpu())
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.title(f"Prediction {i}")
            plt.imshow(torch.argmax(val_outputs, dim=1)[0, :, :, 80].cpu())
            plt.axis("off")

            plt.show()

            if i >= num_images - 1:
                break
