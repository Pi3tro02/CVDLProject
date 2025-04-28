import matplotlib.pyplot as plt

def plot_training_curves(epoch_loss_values, metric_values, val_interval):
    plt.figure("Training Curves", (12, 6))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    x = [i + 1 for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel("Epoch")
    plt.plot(x, y)

    # Plot Dice
    plt.subplot(1, 2, 2)
    plt.title("Validation Mean Dice")
    x = [val_interval * (i + 1) for i in range(len(metric_values))]
    y = metric_values
    plt.xlabel("Epoch")
    plt.plot(x, y)

    plt.show()
    