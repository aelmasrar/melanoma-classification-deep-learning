import torchvision.transforms as transforms


transform_base = transforms.Compose([
    transforms.ToTensor(),
])


def build_transform_normalise(mean, std):
    """Normalisation avec les statistiques calculées sur le jeu d'entraînement."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def build_train_transform_aug(mean, std):
    """Augmentation des données puis normalisation."""
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def build_val_transform(mean, std):
    """Normalisation sans augmentation pour l'évaluation."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
