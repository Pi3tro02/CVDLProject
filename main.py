from config import *
from dataset import get_loaders
from model import get_model
from train_model import train
import torch

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = get_model()
    train_loader, val_loader = get_loaders(TRAIN_DIR, TRAIN_DIR2, BATCH_SIZE, PATCH_SIZE)
    train(model, train_loader, val_loader, device, NUM_EPOCHS, LR)

if __name__ == "__main__":
    main()
