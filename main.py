from config import *
from dataset import get_loaders
from model import get_model
from train_model import train
from check_batch import plot_batch
import torch

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = get_model()
    train_loader, val_loader = get_loaders(TRAIN_DIR, TRAIN_DIR2 , TEST_DIR, BATCH_SIZE, PATCH_SIZE)
    for batch in train_loader:
        print(batch["label"].max(), batch["label"].min(), batch["label"].unique())
        break

    plot_batch(train_loader)
    train(model, train_loader, val_loader, device, NUM_EPOCHS, LR)

if __name__ == "__main__":
    main()
