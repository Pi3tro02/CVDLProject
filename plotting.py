import matplotlib.pyplot as plt

## Function that visualizes and saves two graphs based on the training done
def plot_training_curves(epoch_loss_values, metric_values, val_interval, save_path="training_curves.png"):
    plt.figure("Training Curves", (12, 6))

    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    x = [i + 1 for i in range(len(epoch_loss_values))]
    plt.plot(x, epoch_loss_values, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Validation DICE plot
    plt.subplot(1, 2, 2)
    plt.title("Validation Mean Dice")
    x = [val_interval * (i + 1) for i in range(len(metric_values))]
    plt.plot(x, metric_values, label="Val Dice", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[✓] Grafico salvato come '{save_path}' nella directory del progetto.")
    plt.show()
