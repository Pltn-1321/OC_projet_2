# OC Projet 2 - Fashion Trend Intelligence

Ce projet vise à développer une preuve de concept pour la segmentation automatique de vêtements sur des images à l’aide d’un modèle Hugging Face.

## Structure du projet

```
├── LICENSE            <- Open-source license if one is chosen
├── README.md          <- The top-level README for developers using this project
├── data
│   ├── external       <- Data from third party sources
│   ├── interim        <- Intermediate data that has been transformed
│   ├── processed      <- The final, canonical data sets for modeling
│   └── raw            <- The original, immutable data dump
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
└── src                         <- Source code for this project
    │
    ├── __init__.py             <- Makes src a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    │    
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    ├── plots.py                <- Code to create visualizations 
    │
    └── services                <- Service classes to connect with external platforms, tools, or APIs
        └── __init__.py 
```

--------
## Installation

### 1. Clone le repo et installe Poetry

git clone <URL_DU_REPO>
cd <NOM_DU_REPO>
pip install poetry
poetry install
poetry shell

### 2. Ajoute ton token Hugging Face en toute sécurité
Crée un fichier .env à la racine du projet avec :

HF_API_TOKEN=ton_token_ici
Le fichier .env est déjà listé dans .gitignore pour éviter tout partage accidentel.

### 3. Dépose les données fournies
Place les images d’exemple et leurs annotations dans data/raw/.

### 🚦 Utilisation rapide
Lancer le script d’inférence :

poetry run python src/modeling/predict.py --input data/raw --output data/processed
(à adapter selon le nom réel de ton script)

### Générer des visualisations :

Utilise les notebooks dans le dossier notebooks/ ou le module src/plots.py pour afficher des images originales vs. segmentées.


## Contact
Pierre.pluton@outlook.fr

*Projet réalisé dans le cadre de la formation OpenClassrooms AI Engineer.*
