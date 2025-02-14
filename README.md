# OCP7 - Implémentez un modèle de scoring
Projet n°7 : Développement d’un modèle de scoring de crédit

## Présentation du projet
Ce projet vise à développer un modèle de scoring de crédit capable d’évaluer la probabilité qu’un client rembourse son prêt, même s’il dispose d’un historique de crédit limité ou inexistant.

### L’objectif est double :

- Construire un modèle de machine learning performant pour classer les clients en fonction de leur risque de défaut de paiement.
- Développer un tableau de bord interactif pour garantir la transparence des décisions d’octroi de crédit.

### Missions principales

- Développer un algorithme de classification performant en exploitant des sources de données variées.
- Implémenter des techniques de traitement des données déséquilibrées (undersampling).
- Sélectionner et entraîner plusieurs modèles : Dummy Classifier, Régression Logistique, LightGBM, Random Forest.
- Optimiser les hyperparamètres et ajuster le seuil de classification pour améliorer les performances.
- Déployer une API Flask permettant d’interroger le modèle en temps réel.
- Créer un dashboard Streamlit pour visualiser les prédictions et faciliter l’interprétation des résultats.
- Mettre en place GitHub Actions pour automatiser les tests et l’intégration continue.

## Exploration et Feature Engineering

### Source des données
#### Origine : 
Jeu de données issu de Kaggle.
#### Contenu : 
Un fichier ZIP contenant 10 fichiers CSV, regroupant des informations sur les clients et leurs demandes de crédit.
#### Taille : 
307 511 clients avec 122 variables explicatives (name_contract_type, code_gender, etc.).
#### Label cible : 
target (1 = défaut de paiement, 0 = pas de défaut).

### Caractéristiques des données
Données déséquilibrées :
92% des clients sont réguliers.
8% des clients présentent un défaut de paiement.

### Prétraitement appliqué :
- One-hot encoding des variables catégoriques.
- Détection et gestion des outliers (days_employed contenait des valeurs aberrantes > 1000 ans).
- Imputation des valeurs manquantes (remplacement par la médiane).
- Normalisation avec MinMaxScaler.
- Création de nouvelles features métiers

  ## Méthodologie et Modélisation

### 1. Prétraitement et Séparation des données
- Fusion des différents fichiers (bureau, previous applications, credit card balance, etc.)
- Gestion des valeurs infinies et des caractères spéciaux.
- Suppression des colonnes avec trop de valeurs manquantes.
- Séparation des données en train / test.

### 2. Gestion du déséquilibre des classes
Technique choisie : Undersampling pour réduire la taille de la classe majoritaire et limiter le risque d’overfitting.

### 3. Algorithmes de classification utilisés
#### Baseline : 
Dummy Classifier.
#### Modèles avancés :
- Régression Logistique (avec optimisation du paramètre de régularisation C).
- Random Forest (100 arbres).
- LightGBM (modèle optimisé pour le déséquilibre des classes).

### 4. Optimisation et suivi des expérimentations
- MLflow pour le suivi des expérimentations et des performances.
- GridSearch pour optimiser les hyperparamètres.

### 5. Choix du modèle optimal
#### Métriques utilisées :
Score AUC (0.7638 pour LightGBM).
Business Score basé sur une pondération des erreurs (FN pénalisé 10 fois plus que FP).
Optimisation du seuil de classification pour maximiser le Business Score.

## Déploiement et Industrialisation
### 1. API Flask
- Développement d’une API permettant de faire des prédictions en temps réel.
- Déploiement automatique sur Heroku.
### 2. Dashboard interactif avec Streamlit
Affichage des résultats sous forme de jauges interactives.
Permet aux gestionnaires de crédit d’interpréter facilement les décisions du modèle.
### 3. Intégration Continue et Automatisation
GitHub Actions pour exécuter des tests unitaires et valider le code avant chaque mise en production.
Organisation du code en répertoires structurés pour faciliter la maintenance.

## Technologies utilisées
Langage principal : Python
IDE : VS Code
Librairies : Pandas, NumPy, Scikit-Learn, LightGBM, Flask, Streamlit
Outils de suivi : MLflow
Déploiement : GitHub, GitHub Actions, Heroku

## Installation et Exécution du Projet
### 1. Cloner le projet
bash
Copier
Modifier
git clone https://github.com/mon-repo/OCP7.git
cd OCP7
### 2. Installer les dépendances
bash
Copier
Modifier
pip install -r requirements.txt
### 3. Lancer l’API Flask
bash
Copier
Modifier
python app.py
API accessible à : http://127.0.0.1:5000/predict

### 4. Lancer le Dashboard Streamlit
bash
Copier
Modifier
streamlit run dashboard.py
Dashboard accessible à : http://localhost:8501

🤝 Comment contribuer
Forker le projet.
Créer une branche (feature/ma-nouvelle-feature).
Committer vos modifications (git commit -m "Ajout d'une nouvelle fonctionnalité").
Pousser la branche (git push origin feature/ma-nouvelle-feature).
Créer une Pull Request sur GitHub.
📩 Informations supplémentaires
Auteur : Saholy RATSIMANOHITRA
Date : Février 2025
Entreprise : Prêt à dépenser (spécialisée dans les crédits à la consommation).
✅ Ce README permet :
✅ D’expliquer le contexte du projet et son objectif.
✅ De détailler les étapes clés du modèle de scoring.
✅ De guider l’installation et l’utilisation du projet.
✅ D’illustrer les technologies et outils utilisés.
✅ D’encourager les contributions et la collaboration.

Ce document est maintenant complet, clair et structuré 






# OCP7
Projet n°7 : "Implémentez un modèle de scoring"

Exploration et feature engineering:
Source de données: Kaggle


## Présentatoon du projet

Ce projet vise à développer un modèle de scoring sur des données déséquilibrées, en suivant plusieurs étapes et en utilisant diverses techniques :

- **Traitement des données déséquilibrées**: Application de l'undersampling.
- **Sélection et entraînement de modèles** : Mise en place de modèles comme le dummy régressif, la régression logistique, LightGBM (LGBM) et la forêt aléatoire (Random Forest).
- **Suivi des expérimentations avec MLflow** : Utilisation de MLflow pour documenter et suivre les différentes expériences de modélisation.
- **Choix du meilleur modèle** : Sélection du modèle optimal en fonction des métriques AUC et Business Score, ainsi que du tuning des hyperparamètres.
- **Tuning du seuil de classification** : Ajustement du seuil de classification pour optimiser les performances du modèle retenu.
- **Déploiement d'une API Flask** : Création d'une API Flask pour interroger le modèle de prédiction, mise en production sur Heroku.
- **Mise en place de GitHub et GitHub Actions** : Utilisation de GitHub pour le versioning et l'intégration continue grâce à GitHub Actions, avec des tests unitaires.
- **Développement d'un dashboard interactif** : Création d'un tableau de bord avec Streamlit, incluant des jauges pour visualiser la probabilité de prédiction, destiné aux gestionnaires de relation client.

## Les technologies utilisées
- Python
- VS Code
- MLFlow
- Github, Github actions
- Heroku
  
## Comment installer et exécuter le projet (dépendances, commandes à exécuter)


## Comment contribuer (si d’autres personnes veulent participer)


## Informations supplémentaires (ex: auteur, contact, licence)
