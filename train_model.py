import torch
import monai
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from tqdm import tqdm

class BCEDiceLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(sigmoid=True)  # MONAI DiceLoss

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        return bce_loss + dice_loss

def evaluate(model, val_loader, device, num_classes):
    model.eval()
    dice_metric = DiceMetric(include_background=True, reduction="mean_batch", get_not_nans=False)

    with torch.no_grad():
        for batch_data in val_loader:
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)

            # Se labels hanno più dimensioni (4D), rendili 3D
            if labels.ndim == 4:
                labels = labels.unsqueeze(1)

            labels = (labels > 0).float()

            # Calcola l'output del modello
            outputs = sliding_window_inference(inputs, roi_size=(96, 96, 32), sw_batch_size=1, predictor=model)
            
            # Applica sigmoid e threshold a 0.5 per ottenere la maschera binaria
            outputs = torch.sigmoid(outputs)
            predicted = (outputs > 0.5).float()

            dice_metric(predicted, labels)

        mean_dice = dice_metric.aggregate().mean().item()
        dice_metric.reset()

    return mean_dice

def train(model, train_loader, val_loader, device, num_epochs, lr, save_path="best_model.pth"):
    model.to(device)
    loss_function = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5)
    
    best_val_dice = -1.0  # inizializza il peggiore valore possibile

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        step = 0

        for batch_data in tqdm(train_loader, desc=f"[Epoch {epoch+1}]"):
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            optimizer.zero_grad()

            outputs = model(inputs)

            # Etichetta: fai un threshold per ottenere valori 0 o 1 (binary mask)
            labels = (labels > 0).float()  # Assicurati che le etichette siano binarie

            # La loss non ha bisogno di One Hot Encoding in questo caso
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            step += 1

        avg_loss = epoch_loss / step
        print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")

        # VAL
        val_dice = evaluate(model, val_loader, device, num_classes=1)  # Modifica num_classes a 1 per binario
        print(f"Validation Dice: {val_dice:.4f}")

        scheduler.step(val_dice)

        # SALVA IL MIGLIORE MODELLO
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), save_path)
            print(f"Saved new best model at epoch {epoch+1} with Val Dice {val_dice:.4f}")
