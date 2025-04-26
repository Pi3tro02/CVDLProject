import torch
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from tqdm import tqdm

def train(model, train_loader, val_loader, device, num_epochs, lr):
    model.to(device)
    loss_function = DiceLoss(sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), lr)
    dice_metric = DiceMetric(include_background=True, reduction="mean")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in tqdm(train_loader, desc=f"[Epoch {epoch+1}]"):
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            step += 1
        print(f"Epoch {epoch + 1} average loss: {epoch_loss / step:.4f}")

        # Val
        model.eval()
        dice_metric.reset()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model)
                dice_metric(val_outputs, val_labels)
        print(f"Validation Dice: {dice_metric.aggregate().item():.4f}")
