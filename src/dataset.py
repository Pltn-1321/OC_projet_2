import os
import numpy as np
import cv2


def list_images(image_dir, max_images=None):
    """
    Liste les chemins des images dans un dossier.

    Args:
        image_dir (str): Chemin du dossier contenant les images.
        max_images (int, optional): Nombre maximum d'images à retourner. Si None, retourne toutes les images.
    Returns:
        list: Liste des chemins des images.
    Raises:
        FileNotFoundError: Si le dossier n'existe pas.
        ValueError: Si le dossier n'est pas un répertoire.
        Exception: Si une erreur se produit lors de la lecture du dossier.
    """
    if not os.path.exists(image_dir):
        raise FileNotFoundError(
            f"Le dossier {image_dir} n'existe pas."
        )  # verification de l'existence du dossier
    if not os.path.isdir(image_dir):
        raise ValueError(
            f"{image_dir} n'est pas un répertoire."
        )  # verification que c'est un dossier

    try:
        all_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]
        image_files = [
            f
            for f in all_files
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        ]
        if max_images is not None:
            image_files = image_files[:max_images]
        return image_files
    except Exception as e:
        raise Exception(f"Erreur lors de la lecture du dossier {image_dir} : {e}")


def get_mask_path_from_image(image_path: str, mask_dir: str) -> str:
    """
    À partir du chemin d'une image, génère le chemin du masque cible correspondant.

    Args:
        image_path (str): Chemin complet de l'image (ex: .../IMG/image_45.png)
        mask_dir (str): Chemin du dossier des masques (ex: .../Mask/)

    Returns:
        str: Chemin complet du masque correspondant (ex: .../Mask/mask_45.png)
    """
    basename = os.path.basename(image_path)  # ex: 'image_45.png'
    img_num = basename.split("_")[-1].split(".")[0]  # ex: '45'
    mask_name = f"mask_{img_num}.png"
    mask_path = os.path.join(mask_dir, mask_name)
    return mask_path


def batch_get_true_and_pred_masks(
    image_paths: list[str], mask_dir: str, masks_pred: list[np.ndarray]
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """
    Associe à chaque image son masque de terrain (true) et son masque prédit (pred).

    Args:
        image_paths (list): Chemins des images.
        mask_dir (str): Dossier contenant les masques de terrain.
        masks_pred (list): Liste des masques prédits (même ordre que image_paths).

    Returns:
        list: Liste de tuples (img_path, mask_true, mask_pred)
    """
    out = []
    for i, img_path in enumerate(image_paths):
        mask_true_path = get_mask_path_from_image(img_path, mask_dir)
        mask_true = cv2.imread(mask_true_path, cv2.IMREAD_GRAYSCALE)
        mask_pred = masks_pred[i]
        out.append((img_path, mask_true, mask_pred))
    return out
