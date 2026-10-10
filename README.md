# Classification de mélanomes par Deep Learning

Dans ce projet académique, j'ai travaillé sur la classification d'images de lésions cutanées avec PyTorch. L'objectif était de distinguer les lésions bénignes des mélanomes et de comparer plusieurs approches de Deep Learning.

## Méthodes testées

J'ai commencé par entraîner un CNN à partir de zéro. J'ai travaillé sur la normalisation des images et la data augmentation. J'ai aussi comparé plusieurs optimiseurs et utilisé un learning rate scheduler.

J'ai ensuite testé le transfer learning avec ResNet18, puis le fine-tuning du dernier bloc convolutif et de la couche de classification.

## Résultats

Le meilleur résultat enregistré a été obtenu avec le fine-tuning de ResNet18 :

- **Accuracy : 95,55 %** sur 2 000 images ;
- **Rappel mélanome : 94,70 %** ;
- **1 911 images correctement classées** sur 2 000.

Ces résultats concernent le dossier `melanoma-cancer-dataset/test`, utilisé comme ensemble d'évaluation dans les scripts. Ils ne correspondent donc pas à une validation clinique indépendante.

![Courbes d'entraînement ResNet18](courbes_resnet.png)

![Matrice de confusion](matrice_confusion.png)

## Organisation du projet

Le fichier `train_resnet.py` compare un ResNet18 dont le backbone est gelé avec une version où le dernier bloc est entraîné.

Le fichier `analyze.py` charge le modèle sauvegardé et génère la matrice de confusion et le rapport de classification.

Le jeu de données est organisé ainsi :

```text
melanoma-cancer-dataset/
├── train/
└── test/
```

## Installation et exécution

```bash
pip install -r requirements.txt
python train_resnet.py
python analyze.py
```

## Outils

Python, PyTorch, Torchvision, NumPy, Matplotlib, Pillow et scikit-learn.

Ce projet est une étude académique. Les résultats ne permettent pas d'utiliser le modèle pour un diagnostic médical.
