import torch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.data import DataLoader, Dataset
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ToTensord, Spacingd, Orientationd, ScaleIntensityRanged

from config import TEST_DIR, SAVE_PATH, PATCH_SIZE, BATCH_SIZE
from dataset import get_data_dicts, get_loaders
from model import get_model
from inference import load_model
from visualize_prediction import visualize_prediction

## Preprocessing transformation to images and masks
def get_test_loader(test_dir, patch_size, batch_size):
    test_files = get_data_dicts(test_dir)

    test_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"], a_min=-175, a_max=250,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        ToTensord(keys=["image", "label"]),
    ])

    test_ds = Dataset(data=test_files, transform=test_transforms)
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=2,
    )
    return test_loader

# Sliding window inference for managing big inputs, calculates Dice score
def evaluate_model_on_test(model, test_loader, device):
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_pred = AsDiscrete(threshold=0.5)
    post_label = AsDiscrete(threshold=0.5)

    model.eval()
    with torch.no_grad():
        for data in test_loader:
            inputs = data["image"].to(device)
            print("Shape input:", inputs.shape)
            labels = data["label"].to(device)
            if labels.ndim == 4:
                labels = labels.unsqueeze(1)
            labels = (labels > 0).float()

            outputs = sliding_window_inference(inputs, roi_size=PATCH_SIZE, sw_batch_size=1, predictor=model)
            outputs = torch.sigmoid(outputs)
            preds = post_pred(outputs)
            labs = post_label(labels)

            dice_metric(preds, labs)

        mean_dice = dice_metric.aggregate().mean().item()
        dice_metric.reset()

    return mean_dice

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = get_model()
    model = load_model(model, SAVE_PATH, device)

    # Test loader only
    test_loader = get_test_loader(TEST_DIR, PATCH_SIZE, BATCH_SIZE)

    # Valutazione
    dice = evaluate_model_on_test(model, test_loader, device)
    print(f"Dice score on test set: {dice:.4f}")

    # Visualizzazione predizioni
    print("Visualizing predictions...")
    visualize_prediction(model, test_loader, device, num_samples=3, save_dir="test_predictions")

if __name__ == "__main__":
    main()
