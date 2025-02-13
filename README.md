# OCP7
Projet n°7 : "Implémentez un modèle de scoring"

Exploration et feature engineering:
Source de données: Kaggle

Ce projet vise à développer un modèle de scoring sur des données déséquilibrées, en suivant plusieurs étapes et en utilisant diverses techniques :

- **Traitement des données déséquilibrées**: Application de l'undersampling.
- **Sélection et entraînement de modèles** : Mise en place de modèles comme le dummy régressif, la régression logistique, LightGBM (LGBM) et la forêt aléatoire (Random Forest).
- **Suivi des expérimentations avec MLflow** : Utilisation de MLflow pour documenter et suivre les différentes expériences de modélisation.
- **Choix du meilleur modèle** : Sélection du modèle optimal en fonction des métriques AUC et Business Score, ainsi que du tuning des hyperparamètres.
- **Tuning du seuil de classification** : Ajustement du seuil de classification pour optimiser les performances du modèle retenu.
- **Déploiement d'une API Flask** : Création d'une API Flask pour interroger le modèle de prédiction, mise en production sur Heroku.
- **Mise en place de GitHub et GitHub Actions** : Utilisation de GitHub pour le versioning et l'intégration continue grâce à GitHub Actions, avec des tests unitaires.
- **Développement d'un dashboard interactif** : Création d'un tableau de bord avec Streamlit, incluant des jauges pour visualiser la probabilité de prédiction, destiné aux gestionnaires de relation client.
