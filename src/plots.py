import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def display_segmented_images_batch(original_image_paths, segmentation_masks):
    """
    Affiche côte à côte chaque image originale et son masque de segmentation.

    Args:
        original_image_paths (list of str): Liste des chemins d'accès des images originales à afficher.
        segmentation_masks (list of np.ndarray): Liste des masques segmentés (arrays 2D de même taille que les images originales).

    Raises:
        ValueError:
            - Si les deux listes n'ont pas la même longueur.
            - Si un masque n'est pas un tableau numpy 2D valide.
        FileNotFoundError:
            - Si un fichier image n'existe pas.
        Exception:
            - Si une image ne peut pas être ouverte.
    Returns:
        None. Affiche les paires image originale / masque dans des fenêtres matplotlib.
    """
    if len(original_image_paths) != len(segmentation_masks):
        raise ValueError(
            f"Nombre d'images ({len(original_image_paths)}) différent du nombre de masques ({len(segmentation_masks)})."
        )
    for i, (image_path, mask) in enumerate(
        zip(original_image_paths, segmentation_masks)
    ):
        if mask is None:
            print(f"[{i + 1}] Pas de masque pour l'image {image_path}.")
            continue
        if not isinstance(mask, np.ndarray) or len(mask.shape) != 2:
            raise ValueError(
                f"Le masque #{i + 1} pour l'image '{image_path}' n'est pas un array numpy 2D valide."
            )
        try:
            original_image = Image.open(image_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"L'image '{image_path}' n'existe pas.")
        except Exception as e:
            raise Exception(f"Impossible d'ouvrir l'image '{image_path}' : {e}")
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(original_image)
        plt.title(f"Image originale {i + 1}")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(mask, cmap="jet", alpha=0.5)
        plt.title(f"Masque segmenté {i + 1}")
        plt.axis("off")
        plt.show()
