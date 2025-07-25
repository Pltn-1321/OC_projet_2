import requests
import time
from features import get_image_dimensions, create_masks
import numpy as np


def segment_image(image_path, api_url, api_token) -> np.ndarray:
    """
    Appelle l'API Hugging Face pour segmenter une image.

    Args:
        image_path (str): Chemin de l'image à segmenter.
        api_url (str): URL du modèle d'inférence.
        api_token (str): Token d'API Hugging Face.

    Returns:
        np.ndarray: Masque de segmentation.
    """
    headers = {"Authorization": f"Bearer {api_token}"}
    with open(image_path, "rb") as f:
        image_data = f.read()
    content_type = (
        "image/jpeg" if image_path.lower().endswith((".jpg", ".jpeg")) else "image/png"
    )
    headers["Content-Type"] = content_type

    response = requests.post(api_url, headers=headers, data=image_data)
    if response.status_code != 200:
        print(f"Erreur API : {response.status_code} - {response.text}")
        return None
    results = response.json()
    width, height = get_image_dimensions(image_path)
    mask = create_masks(results, width, height)
    return mask


def segment_images_batch(
    image_paths: list[str], api_url: str, api_token: str, sleep: int = 2
) -> list[np.ndarray]:
    """
    Segmente plusieurs images en appelant l'API pour chacune.

    Args:
        image_paths (list): Chemins des images à segmenter.
        api_url (str): URL du modèle d'inférence.
        api_token (str): Token d'API.
        sleep (int): Pause entre deux appels API.

    Returns:
        list: Liste de masques.
    """
    from tqdm import tqdm

    masks = []
    for img_path in tqdm(image_paths, desc="Batch segmentation"):
        mask = segment_image(img_path, api_url, api_token)
        masks.append(mask)
        time.sleep(sleep)
    return masks


## Single version of compute_iou


def compute_iou(
    mask_true: np.ndarray, mask_pred: np.ndarray, ignore_class: int = 0
) -> tuple[float, dict[int, float]]:
    """
    Calcule le score moyen d'IoU pour chaque classe, hors 'ignore_class'.
    Args:
        mask_true (np.ndarray): masque de vérité terrain.
        mask_pred (np.ndarray): masque prédit.
        ignore_class (int): classe à ignorer (souvent 0 pour background).
    Returns:
        iou_mean (float): score moyen d'IoU.
        iou_per_class (dict[int, float]): IoU pour chaque classe.
    """
    classes = np.unique(mask_true)
    ious = {}
    for class_id in classes:
        if class_id == ignore_class:
            continue
        mask_true_bin = mask_true == class_id
        mask_pred_bin = mask_pred == class_id
        intersection = np.logical_and(mask_true_bin, mask_pred_bin).sum()
        union = np.logical_or(mask_true_bin, mask_pred_bin).sum()
        if union == 0:
            iou = float("nan")  # Classe absente dans l'image
        else:
            iou = intersection / union
        ious[class_id] = iou
    iou_mean = np.nanmean(list(ious.values()))
    return iou_mean, ious


## Batch version of compute_iou


def compute_iou_batch(
    masks_true: list[np.ndarray], masks_pred: list[np.ndarray], ignore_class: int = 0
) -> tuple[list[float], list[dict[int, float]]]:
    """
    Calcule les scores IoU pour chaque paire de masques (batch).

    Args:
        masks_true (list[np.ndarray]): Liste des masques de vérité terrain.
        masks_pred (list[np.ndarray]): Liste des masques prédits.
        ignore_class (int): Classe à ignorer.

    Returns:
        tuple:
            - list[float]: Moyenne IoU pour chaque image.
            - list[dict[int, float]]: Détail IoU par classe pour chaque image.
    """
    iou_means = []
    iou_per_class_list = []
    for mask_true, mask_pred in zip(masks_true, masks_pred):
        iou_mean, iou_per_class = compute_iou(
            mask_true, mask_pred, ignore_class=ignore_class
        )
        iou_means.append(iou_mean)
        iou_per_class_list.append(iou_per_class)
    return iou_means, iou_per_class_list
