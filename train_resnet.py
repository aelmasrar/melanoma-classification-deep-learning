"""Transfer learning et fine-tuning avec ResNet18.

Deux stratégies sont comparées :
- entraînement de la couche de classification avec le backbone gelé ;
- fine-tuning du dernier bloc convolutionnel et de la couche de classification.
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset import MelanomaDataset
from transforms import build_train_transform_aug, build_val_transform
from model import charger_resnet_gele, charger_resnet_finetune, compter_parametres
from train import train_one_epoch, evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device :", device)

TRAIN_DIR = "melanoma-cancer-dataset/train"
VAL_DIR   = "melanoma-cancer-dataset/test"
NUM_EPOCHS = 20
SEED = 42

# Normalisation ImageNet utilisée par le modèle pré-entraîné
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = build_train_transform_aug(IMAGENET_MEAN, IMAGENET_STD)
val_transform = build_val_transform(IMAGENET_MEAN, IMAGENET_STD)

train_dataset = MelanomaDataset(TRAIN_DIR, transform=train_transform)
val_dataset   = MelanomaDataset(VAL_DIR,   transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=0)

num_classes = len(train_dataset.classes)
criterion   = nn.CrossEntropyLoss()

print(f"Taille train : {len(train_dataset)}")
print(f"Taille val   : {len(val_dataset)}")


def run_training(model, optimizer, label):
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    print(f"\n=== {label} ===")
    compter_parametres(model)

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, True)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, device)
        duree = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"  Epoch {epoch:2d}/{NUM_EPOCHS} | "
            f"Loss train {train_loss:.4f} | Loss val {val_loss:.4f} | "
            f"Acc train {train_acc:.4f} | Acc val {val_acc:.4f} | {duree:.1f}s"
        )

    return history


torch.manual_seed(SEED)
resnet_gele = charger_resnet_gele(num_classes, device)
optimizer_gele = optim.Adam(resnet_gele.fc.parameters(), lr=1e-3)

history_gele = run_training(resnet_gele, optimizer_gele, "ResNet18 backbone gelé")

torch.manual_seed(SEED)
resnet_ft = charger_resnet_finetune(num_classes, device)
optimizer_ft = optim.Adam([
    {"params": resnet_ft.layer4.parameters(), "lr": 1e-4},
    {"params": resnet_ft.fc.parameters(),     "lr": 1e-3},
])

history_ft = run_training(resnet_ft, optimizer_ft, "Fine-tuning layer4 + FC")

epochs = range(1, NUM_EPOCHS + 1)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(epochs, history_gele["val_loss"], label="Backbone gelé")
axes[0].plot(epochs, history_ft["val_loss"],   label="Fine-tuning")
axes[0].set_title("Perte de validation")
axes[0].set_xlabel("Époque")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(epochs, history_gele["val_acc"], label="Backbone gelé")
axes[1].plot(epochs, history_ft["val_acc"],   label="Fine-tuning")
axes[1].set_title("Précision de validation")
axes[1].set_xlabel("Époque")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].grid(True)

plt.suptitle("Transfer learning vs Fine-tuning (ResNet18)")
plt.tight_layout()
plt.savefig("courbes_resnet.png")
plt.show()
print("Courbes sauvegardees -> courbes_resnet.png")

torch.save({
    "model_type":       "resnet18_finetune",
    "model_state_dict": resnet_ft.state_dict(),
    "num_classes":      num_classes,
    "classes":          train_dataset.classes,
    "mean":             IMAGENET_MEAN,
    "std":              IMAGENET_STD,
}, "checkpoint_resnet_ft.pth")
print("Modele sauvegarde -> checkpoint_resnet_ft.pth")
