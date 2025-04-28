import torch
import monai
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from tqdm import tqdm
from config import LR, VAL_INTERVAL  # importa anche VAL_INTERVAL

class WeightedBCEDiceLoss(torch.nn.Module):
    def __init__(self, bce_weight=0.3, dice_weight=0.7):
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(sigmoid=True)

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

def evaluate(model, val_loader, device, num_classes):
    model.eval()
    dice_metric = DiceMetric(include_background=True, reduction="mean_batch", get_not_nans=False)

    with torch.no_grad():
        for batch_data in val_loader:
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)

            if labels.ndim == 4:
                labels = labels.unsqueeze(1)

            labels = (labels > 0).float()

            outputs = sliding_window_inference(inputs, roi_size=(128, 128, 64), sw_batch_size=1, predictor=model)
            outputs = torch.sigmoid(outputs)
            predicted = (outputs > 0.5).float()

            dice_metric(predicted, labels)

        mean_dice = dice_metric.aggregate().mean().item()
        dice_metric.reset()

    return mean_dice

def train(model, train_loader, val_loader, device, num_epochs, lr, save_path="best_model.pth"):
    model.to(device)
    loss_function = WeightedBCEDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5)

    best_val_dice = -1.0  # inizializza il peggiore valore possibile

    # LISTE PER SALVARE I VALORI PER IL PLOTTING
    epoch_loss_values = []
    metric_values = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        step = 0

        for batch_data in tqdm(train_loader, desc=f"[Epoch {epoch+1}]"):
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            labels = (labels > 0).float()  # binarizza

            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            step += 1

        avg_loss = epoch_loss / step
        print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")
        epoch_loss_values.append(avg_loss)  # <-- salva la loss media dell'epoch

        # VALIDAZIONE OGNI VAL_INTERVAL
        if (epoch + 1) % VAL_INTERVAL == 0:
            val_dice = evaluate(model, val_loader, device, num_classes=1)
            print(f"Validation Dice: {val_dice:.4f}")
            metric_values.append(val_dice)  # <-- salva la metrica

            scheduler.step(val_dice)

            # SALVA IL MIGLIOR MODELLO
            if val_dice > best_val_dice:
                best_val_dice = val_dice
                torch.save(model.state_dict(), save_path)
                print(f"Saved new best model at epoch {epoch+1} with Val Dice {val_dice:.4f}")

    return epoch_loss_values, metric_values  # <-- ritorna
    