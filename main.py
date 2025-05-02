from config import *
from dataset import get_loaders
from model import get_model
from train_model import train
from check_batch import plot_batch
from plotting import plot_training_curves
from inference import load_model, visualize_inference
from visualize_prediction import visualize_prediction

import torch
import os

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = get_model()
    train_loader, val_loader = get_loaders(TRAIN_DIR, TRAIN_DIR2 , TEST_DIR, BATCH_SIZE, PATCH_SIZE)
    for batch in train_loader:
        print(batch["label"].max(), batch["label"].min(), batch["label"].unique())
        break
    plot_batch(train_loader)

    epoch_loss_values, metric_values = train(model, train_loader, val_loader, device, NUM_EPOCHS, LR, save_path=SAVE_PATH)

    plot_training_curves(epoch_loss_values, metric_values, VAL_INTERVAL)

    # 6. Ricarica best model
    model_path = SAVE_PATH
    model = load_model(model, model_path, device)

    # 7. Visualizza predizione
    visualize_prediction(model, val_loader, device, num_samples=3, save_dir="visualize_prediction.png")

if __name__ == "__main__":
    main()
