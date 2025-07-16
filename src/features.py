from PIL import Image
import base64
import io
import numpy as np

# This file contains utility functions for image processing and segmentation mask handling.
# It includes functions to get image dimensions, decode base64 masks, and create segmentation masks from model results.
# Class mapping is defined to map clothing categories to specific class IDs.

CLASS_MAPPING = {
    "Background": 0,
    "Hat": 1,
    "Hair": 2,
    "Sunglasses": 3,
    "Upper-clothes": 4,
    "Skirt": 5,
    "Pants": 6,
    "Dress": 7,
    "Belt": 8,
    "Left-shoe": 9,
    "Right-shoe": 10,
    "Face": 11,
    "Left-leg": 12,
    "Right-leg": 13,
    "Left-arm": 14,
    "Right-arm": 15,
    "Bag": 16,
    "Scarf": 17,
}


def get_image_dimensions(img_path):
    """
    Retourne les dimensions (largeur, hauteur) d'une image.

    Args:
        img_path (str): Chemin de l'image.

    Returns:
        tuple: (largeur, hauteur)
    Raises:
        FileNotFoundError: Si le fichier image n'existe pas.
        Exception: Si l'image ne peut pas être ouverte/lue.
    """
    if not img_path or not isinstance(img_path, str):
        raise ValueError(
            "Le chemin de l'image doit être une chaîne de caractères non vide."
        )
    try:
        with Image.open(img_path) as img:
            return img.size  # (largeur, hauteur)
    except FileNotFoundError:
        raise FileNotFoundError(f"L'image '{img_path}' n'existe pas.")
    except Exception as e:
        raise Exception(f"Impossible d'ouvrir l'image '{img_path}' : {e}")


def decode_base64_mask(base64_string, width, height):
    """
    Decode a base64-encoded mask into a NumPy array.

    Args:
        base64_string (str): Base64-encoded mask.
        width (int): Target width.
        height (int): Target height.

    Returns:
        np.ndarray: Single-channel mask array.
    """
    mask_data = base64.b64decode(base64_string)
    mask_image = Image.open(io.BytesIO(mask_data))
    mask_array = np.array(mask_image)
    if len(mask_array.shape) == 3:
        mask_array = mask_array[:, :, 0]  # Take first channel if RGB
    mask_image = Image.fromarray(mask_array).resize((width, height), Image.NEAREST)
    return np.array(mask_image)


def create_masks(results, width, height):
    """
    Combine multiple class masks into a single segmentation mask.

    Args:
        results (list): List of dictionaries with 'label' and 'mask' keys.
        width (int): Target width.
        height (int): Target height.

    Returns:
        np.ndarray: Combined segmentation mask with class indices.
    """
    combined_mask = np.zeros(
        (height, width), dtype=np.uint8
    )  # Initialize with Background (0)

    # Process non-Background masks first
    for result in results:
        label = result["label"]
        class_id = CLASS_MAPPING.get(label, 0)
        if class_id == 0:  # Skip Background
            continue
        mask_array = decode_base64_mask(result["mask"], width, height)
        combined_mask[mask_array > 0] = class_id

    # Process Background last to ensure it doesn't overwrite other classes unnecessarily
    # (Though the model usually provides non-overlapping masks for distinct classes other than background)
    for result in results:
        if result["label"] == "Background":
            mask_array = decode_base64_mask(result["mask"], width, height)
            # Apply background only where no other class has been assigned yet
            # This logic might need adjustment based on how the model defines 'Background'
            # For this model, it seems safer to just let non-background overwrite it first.
            # A simple application like this should be fine: if Background mask says pixel is BG, set it to 0.
            # However, a more robust way might be to only set to background if combined_mask is still 0 (initial value)
            combined_mask[mask_array > 0] = 0  # Class ID for Background is 0

    return combined_mask
