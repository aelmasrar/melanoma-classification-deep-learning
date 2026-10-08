# Melanoma Classification with Deep Learning

Projet de classification d'images de lésions cutanées avec PyTorch. L'objectif est de comparer plusieurs approches de Deep Learning pour distinguer les images bénignes des mélanomes.

## Approches étudiées

- CNN simple entraîné from scratch
- normalisation des images
- data augmentation
- comparaison de plusieurs optimiseurs
- learning rate scheduler
- transfer learning avec ResNet18
- fine-tuning de ResNet18

## Organisation

Les différents scripts permettent de tester progressivement les choix d'entraînement et de comparer leurs performances. `analyze.py` permet ensuite d'analyser les prédictions avec une matrice de confusion et des exemples d'images bien ou mal classées.

Le dataset utilisé par les scripts est organisé dans `melanoma-cancer-dataset/train` et `melanoma-cancer-dataset/test`.

## Technologies

Python, PyTorch, Torchvision, NumPy, Matplotlib et scikit-learn.
