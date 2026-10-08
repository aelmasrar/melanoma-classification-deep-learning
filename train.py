import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


def calculer_mean_std(dataset):
    """Calcule la moyenne et l'écart-type de chaque canal sur le train set."""
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    channel_sum = torch.zeros(3)
    channel_sum_sq = torch.zeros(3)
    num_pixels = 0

    for images, _ in loader:
        channel_sum += images.sum(dim=(0, 2, 3))
        channel_sum_sq += (images ** 2).sum(dim=(0, 2, 3))
        num_pixels += images.shape[0] * images.shape[2] * images.shape[3]

    mean = channel_sum / num_pixels
    variance = channel_sum_sq / num_pixels - mean ** 2
    std = torch.sqrt(torch.clamp(variance, min=0.0))

    print(f"MEAN : {mean.tolist()}")
    print(f"STD  : {std.tolist()}")

    return mean.tolist(), std.tolist()


def train_one_epoch(model, loader, criterion, optimizer, device, freeze_frozen_bn=False):
    """Effectue une epoch d'entraînement. Retourne (loss_moyenne, accuracy)."""
    model.train()

    # Les BatchNorm des parties gelées gardent leurs statistiques pré-entraînées.
    if freeze_frozen_bn:
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                if all(not p.requires_grad for p in module.parameters()):
                    module.eval()

    total_loss, total_correct, total_samples = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss    += loss.item() * images.size(0)
        predictions    = outputs.argmax(dim=1)
        total_correct += (predictions == labels).sum().item()
        total_samples += images.size(0)

    return total_loss / total_samples, total_correct / total_samples


def evaluate(model, loader, criterion, device):
    """Évalue le modèle sur un loader. Retourne (loss_moyenne, accuracy)."""
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss    += loss.item() * images.size(0)
            predictions    = outputs.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += images.size(0)

    return total_loss / total_samples, total_correct / total_samples


def tracer_courbes(history, titre="CNN", save_name=None):
    """Trace les courbes de loss et d'accuracy train/val."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Train",      color="steelblue")
    ax1.plot(epochs, history["val_loss"],   label="Validation", color="tomato")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title(f"Loss — {titre}"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, history["train_acc"], label="Train",      color="steelblue")
    ax2.plot(epochs, history["val_acc"],   label="Validation", color="tomato")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_title(f"Accuracy — {titre}"); ax2.legend(); ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    if save_name:
        plt.savefig(save_name, dpi=150)
    plt.show()
