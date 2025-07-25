from PIL import Image
import base64
import io
import numpy as np
from typing import Tuple, List, Dict, Any

# Dictionnaire de correspondance entre les noms des classes et leurs identifiants numériques
# Utilisé pour convertir les labels textuels en valeurs numériques dans le masque final
CLASS_MAPPING = {
    "Background": 0,  # Arrière-plan
    "Hat": 1,  # Chapeau
    "Hair": 2,  # Cheveux
    "Sunglasses": 3,  # Lunettes de soleil
    "Upper-clothes": 4,  # Vêtements du haut
    "Skirt": 5,  # Jupe
    "Pants": 6,  # Pantalon
    "Dress": 7,  # Robe
    "Belt": 8,  # Ceinture
    "Left-shoe": 9,  # Chaussure gauche
    "Right-shoe": 10,  # Chaussure droite
    "Face": 11,  # Visage
    "Left-leg": 12,  # Jambe gauche
    "Right-leg": 13,  # Jambe droite
    "Left-arm": 14,  # Bras gauche
    "Right-arm": 15,  # Bras droit
    "Bag": 16,  # Sac
    "Scarf": 17,  # Écharpe
}


def get_image_dimensions(img_path: str) -> Tuple[int, int]:
    """
    Récupère les dimensions d'une image.

    Cette fonction ouvre une image et retourne sa largeur et hauteur.
    Utile pour dimensionner correctement les masques de segmentation.

    Args:
        img_path (str): Chemin vers le fichier image

    Returns:
        Tuple[int, int]: (largeur, hauteur) en pixels

    Raises:
        ValueError: Si le chemin est invalide
        FileNotFoundError: Si le fichier n'existe pas
        Exception: Si l'image ne peut pas être lue
    """
    if not img_path or not isinstance(img_path, str):
        raise ValueError("Le chemin de l'image doit être une chaîne non vide")

    try:
        with Image.open(img_path) as img:
            return img.size  # PIL retourne (largeur, hauteur)
    except FileNotFoundError:
        raise FileNotFoundError(f"Image introuvable : {img_path}")
    except Exception as e:
        raise Exception(f"Erreur lors de l'ouverture de {img_path} : {e}")


def decode_base64_mask(base64_string: str, width: int, height: int) -> np.ndarray:
    """
    Décode un masque encodé en base64 et le redimensionne.

    Les API de segmentation retournent souvent les masques sous forme de chaînes base64.
    Cette fonction les convertit en arrays NumPy utilisables.

    Args:
        base64_string (str): Masque encodé en base64
        width (int): Largeur cible du masque
        height (int): Hauteur cible du masque

    Returns:
        np.ndarray: Masque sous forme d'array 2D (hauteur, largeur)
    """
    # Décode la chaîne base64 en données binaires
    mask_data = base64.b64decode(base64_string)

    # Convertit les données binaires en image PIL
    mask_image = Image.open(io.BytesIO(mask_data))

    # Convertit en array NumPy
    mask_array = np.array(mask_image)

    # Si l'image est en couleur (RGB), prend seulement le premier canal
    if len(mask_array.shape) == 3:
        mask_array = mask_array[:, :, 0]

    # Redimensionne le masque aux dimensions voulues
    # NEAREST évite l'interpolation qui pourrait créer des valeurs intermédiaires
    mask_image = Image.fromarray(mask_array).resize((width, height), Image.NEAREST)

    return np.array(mask_image)


def create_masks(results: List[Dict[str, Any]], width: int, height: int) -> np.ndarray:
    """
    Combine plusieurs masques de classes en un seul masque de segmentation.

    Cette fonction prend les résultats de l'API (une liste de masques par classe)
    et les combine en un seul masque où chaque pixel a la valeur de sa classe.

    Args:
        results (List[Dict[str, Any]]): Liste de dictionnaires avec les clés:
                                       - 'label': nom de la classe (str)
                                       - 'mask': masque base64 de cette classe (str)
        width (int): Largeur du masque final
        height (int): Hauteur du masque final

    Returns:
        np.ndarray: Masque combiné où chaque pixel contient l'ID de sa classe
                   (0 = Background, 1 = Hat, 2 = Hair, etc.)
    """
    # Initialise un masque vide avec tous les pixels en arrière-plan (classe 0)
    combined_mask = np.zeros((height, width), dtype=np.uint8)

    # Traite d'abord toutes les classes sauf le Background
    # Cela évite que l'arrière-plan écrase les autres classes
    for result in results:
        label = result["label"]
        class_id = CLASS_MAPPING.get(label, 0)  # 0 par défaut si classe inconnue

        if class_id == 0:  # Skip Background pour l'instant
            continue

        # Décode le masque de cette classe
        mask_array = decode_base64_mask(result["mask"], width, height)

        # Applique cette classe aux pixels où le masque est actif (> 0)
        combined_mask[mask_array > 0] = class_id

    # Traite le Background en dernier
    # Le Background ne devrait écraser les autres classes que si nécessaire
    for result in results:
        if result["label"] == "Background":
            mask_array = decode_base64_mask(result["mask"], width, height)
            # Applique le Background (classe 0) aux pixels marqués dans son masque
            combined_mask[mask_array > 0] = 0

    return combined_mask


# --- Palette couleur (colormap) pour chaque classe ---
custom_colormap = {
    0: (0, 0, 0),  # Background - Noir (à ajouter si tu veux aussi afficher le fond)
    1: (0, 255, 255),  # Jaune - Hat
    2: (0, 165, 255),  # Orange - Hair
    3: (255, 0, 255),  # Magenta - Sunglasses
    4: (0, 0, 255),  # Rouge - Upper-clothes
    5: (255, 255, 0),  # Cyan - Skirt
    6: (0, 255, 0),  # Vert - Pants
    7: (255, 0, 0),  # Bleu - Dress
    8: (128, 0, 128),  # Violet - Belt
    9: (0, 255, 255),  # Jaune - Left-shoe
    10: (255, 140, 0),  # Orange foncé - Right-shoe
    11: (200, 180, 140),  # Beige - Face
    12: (200, 180, 140),  # Beige - Left-leg
    13: (200, 180, 140),  # Beige - Right-leg
    14: (200, 180, 140),  # Beige - Left-arm
    15: (200, 180, 140),  # Beige - Right-arm
    16: (0, 128, 255),  # Bleu clair - Bag
    17: (255, 20, 147),  # Rose - Scarf
}

legend_labels = {
    "0": "Background",
    "1": "Hat",
    "2": "Hair",
    "3": "Sunglasses",
    "4": "Upper-clothes",
    "5": "Skirt",
    "6": "Pants",
    "7": "Dress",
    "8": "Belt",
    "9": "Left-shoe",
    "10": "Right-shoe",
    "11": "Face",
    "12": "Left-leg",
    "13": "Right-leg",
    "14": "Left-arm",
    "15": "Right-arm",
    "16": "Bag",
    "17": "Scarf",
}


def colorize_mask(mask: np.ndarray, colormap: dict) -> np.ndarray:
    """
    Colorie un masque 2D de labels selon le colormap fourni.
    Args:
        mask (np.ndarray): Masque 2D où chaque pixel contient l'ID de la classe
        colormap (dict): Dictionnaire {ID classe: (R, G, B)}
    Returns:
        np.ndarray: Image couleur (H, W, 3) prête à afficher
    """
    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in colormap.items():
        color_mask[mask == class_id] = color
    return color_mask
