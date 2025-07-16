Bien sûr Pierre, voici une version **améliorée, claire et “pro”** de ton README, **à jour avec ta structure factorisée et l’usage de Poetry, python-dotenv et modules maison**.
👉 Prêt à copier-coller dans Notion, GitHub, ou ailleurs !

---

# OC Projet 2 — Fashion Trend Intelligence

Ce projet est une preuve de concept de **segmentation automatique de vêtements sur images** via un modèle Hugging Face.
Le code est entièrement factorisé et modulaire pour une meilleure maintenance et réutilisation, selon les standards AI Engineer OpenClassrooms.

---

## 📁 **Structure du projet**

```plaintext
├── LICENSE
├── README.md
├── data/
│   ├── external/
│   ├── interim/
│   ├── processed/
│   └── raw/
├── models/
├── notebooks/
├── references/
├── reports/
│   └── figures/
├── requirements.txt   # (généré automatiquement si besoin)
├── pyproject.toml     # (Poetry)
├── .env               # (token API, jamais versionné)
└── src/
    ├── __init__.py
    ├── config.py
    ├── dataset.py
    ├── features.py
    ├── plots.py
    ├── modeling/
    │   ├── __init__.py
    │   ├── predict.py
    │   └── train.py
    └── services/
        └── __init__.py
```

---

## 🚀 **Installation & setup**

### 1. **Cloner le repo & installer Poetry**

```bash
git clone <URL_DU_REPO>
cd <NOM_DU_REPO>
pip install poetry
poetry install
poetry shell
```

---

### 2. **Configuration de la clé API**

- Crée un fichier `.env` à la racine du projet :

  ```dotenv
  API_TOKEN=ton_token_huggingface_ici
  ```

- Le fichier `.env` est **déjà listé dans `.gitignore`**.

---

### 3. **Dépose tes données**

- Place les images d’exemple dans `data/raw/`.

---

### 4. **Lancer un pipeline d’inférence dans un notebook**

Dans un nouveau notebook du dossier `notebooks/` :

```python
import sys
sys.path.append("../src")

from dataset import list_images
from features import get_image_dimensions, decode_base64_mask, create_masks
from modeling.predict import segment_images_batch
from plots import display_segmented_images_batch

from dotenv import load_dotenv
import os

load_dotenv("../.env")
api_token = os.getenv("API_TOKEN")

image_dir = "../data/raw/top_influenceurs_2024/IMG/"
max_images = 2
api_url = "https://router.huggingface.co/hf-inference/models/sayeed99/segformer_b3_clothes"

image_paths = list_images(image_dir, max_images)
batch_seg_results = segment_images_batch(image_paths, api_url, api_token)
display_segmented_images_batch(image_paths, batch_seg_results)
```

---

### 5. **(Option) Générer un fichier requirements.txt**

Pour compatibilité universelle (hors Poetry) :

```bash
poetry export --format=requirements.txt --output=requirements.txt
```

---

## 💡 **Bonnes pratiques**

- **Tous les modules réutilisables sont factorisés dans `/src`** pour faciliter la maintenance, le test, et la reproductibilité.
- **Pas d’API key dans le code :** utilisez le `.env` et `python-dotenv` pour la sécurité.
- **Le notebook sert uniquement de pipeline, pas d’implémentation brute.**
- **Redémarrez le kernel Jupyter** après modification de la structure du projet ou du `.env`.

---

## 🧑‍💻 **Contact**

> [pierre.pluton@outlook.fr](mailto:pierre.pluton@outlook.fr) > _Projet réalisé dans le cadre de la formation OpenClassrooms AI Engineer._

---
