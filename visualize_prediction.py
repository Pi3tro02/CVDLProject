import os
import torch
import matplotlib.pyplot as plt
from monai.inferers import sliding_window_inference

def visualize_prediction(model, dataloader, device, num_samples=3, save_dir=None):
    model.eval()
    count = 0

    os.makedirs(save_dir, exist_ok=True) if save_dir else None

    with torch.no_grad():
        for batch_data in dataloader:
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)

            # Prediction via sliding window
            outputs = sliding_window_inference(inputs, roi_size=(128, 128, 64), sw_batch_size=1, predictor=model)
            outputs = torch.sigmoid(outputs)
            preds = (outputs > 0.5).float()

            for i in range(inputs.shape[0]):
                image = inputs[i, 0].cpu()
                label = labels[i, 0].cpu()
                pred = preds[i, 0].cpu()

                mid_slice = image.shape[-1] // 2  # Central slice in z axis

                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                axs[0].imshow(image[:, :, mid_slice], cmap="gray")
                axs[0].set_title("Image")
                axs[1].imshow(label[:, :, mid_slice], cmap="Reds", alpha=0.6)
                axs[1].set_title("Ground Truth")
                axs[2].imshow(pred[:, :, mid_slice], cmap="Blues", alpha=0.6)
                axs[2].set_title("Prediction")

                for ax in axs:
                    ax.axis("off")

                plt.tight_layout()

                if save_dir:
                    fname = f"sample_{count}_slice_{mid_slice}.png"
                    path = os.path.join(save_dir, fname)
                    plt.savefig(path)
                    print(f"Saved visualization to {path}")
                else:
                    plt.show()

                plt.close()
                count += 1
                if count >= num_samples:
                    return
