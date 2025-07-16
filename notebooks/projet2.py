import os
import requests
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm.notebook import tqdm
import base64
import io
import sys  # noqa: F401
import time


image_dir = "../data/raw/top_influenceurs_2024/IMG/"  # Dossier contenant les images

max_images = 3  # Nombre maximum d'images à traiter

api_token = os.getenv("HF_API_TOKEN")
if not api_token:
    raise ValueError(
        "Veuillez définir la variable d'environnement 'HF_API_TOKEN' avec votre token Hugging Face."
    )

API_URL = (
    "https://router.huggingface.co/hf-inference/models/sayeed99/segformer_b3_clothes"
)
headers = {
    "Authorization": f"Bearer {api_token}"
    # Le "Content-Type" sera ajouté dynamiquement lors de l'envoi de l'image
}

# On liste tous les fichiers du dossier, avec leur chemin complet
all_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]

# On garde seulement les fichiers image valides (optionnel)
image_files = [
    f for f in all_files if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
]

image_paths = image_files[:max_images]  # On limite le nombre d'images à traiter

# Vérification si des images sont présentes
if not image_paths:
    print(f"Aucune image trouvée dans '{image_dir}'. Veuillez y ajouter des images.")
else:
    print(f"{len(image_paths)} image(s) à traiter : {image_paths}")


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
    Get the dimensions of an image.

    Args:
        img_path (str): Path to the image.

    Returns:
        tuple: (width, height) of the image.
    """
    original_image = Image.open(img_path)
    return original_image.size


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


if image_paths:
    single_image_path = image_paths[0]
    print(f"Traitement de l'image : {single_image_path}")

    try:
        # Ouvre et lit l'image en mode binaire
        with open(single_image_path, "rb") as f:
            image_data = f.read()
        print("Image lue avec succès.")

        # Affiche les dimensions de l'image
        dimension_image = get_image_dimensions(single_image_path)
        print("Dimensions de l'image :", dimension_image)

        # Déterminer le `Content-Type` (par exemple, `"image/jpeg"` ou `"image/png"`).
        content_type = (
            "image/jpeg"
            if single_image_path.lower().endswith(".jpg")
            or single_image_path.lower().endswith(".jpeg")
            else "image/png"
        )
        headers["Content-Type"] = content_type
        print("En-têtes de la requête :", headers)
        print("Envoi de l'image à l'API...")
        # Envoie l'image à l'API
        response = requests.post(API_URL, headers=headers, data=image_data)
        print("Réponse de l'API reçue.")
        if response.status_code != 200:
            print(
                f"Erreur lors de l'appel à l'API : {response.status_code} - {response.text}"
            )
            raise Exception("L'appel à l'API a échoué.")
        print("Traitement de l'image terminé.")
        # Traite la réponse de l'API
        results = response.json()
        print("Résultats reçus de l'API.")
        # Affiche les résultats
        print("Résultats de la segmentation :", results)
        # Crée les masques à partir des résultats
        mask = create_masks(results, dimension_image[0], dimension_image[1])
        print("Masque créé avec succès.")
        # Vérifie si le masque a été créé correctement
        if mask is None or mask.size == 0:
            raise Exception(
                "Le masque de segmentation est vide ou n'a pas été créé correctement."
            )
        # Afficher l'image originale et le masque segmenté.
        original_image = Image.open(single_image_path)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(original_image)
        plt.title("Image originale")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(mask, cmap="jet", alpha=0.5)
        plt.title("Masque de segmentation")
        plt.axis("off")
        plt.show()
        print("Image originale et masque affichés avec succès.")

    except Exception as e:
        print(f"Une erreur est survenue : {e}")
else:
    print(
        "Aucune image à traiter. Vérifiez la configuration de 'image_dir' et 'max_images'."
    )


# Note: The code above is designed to process a single image at a time.


def segment_images_batch(list_of_image_paths):
    """
    Segmente une liste d'images en utilisant l'API Hugging Face.

    Args:
        list_of_image_paths (list): Liste des chemins vers les images.

    Returns:
        list: Liste des masques de segmentation (tableaux NumPy).
              Contient None si une image n'a pas pu être traitée.
    """
    batch_segmentations = []

    for image_path in tqdm(list_of_image_paths, desc="Traitement des images en batch"):
        try:
            # Ouvre et lit l'image en mode binaire
            with open(image_path, "rb") as f:
                image_data = f.read()

            # Affiche les dimensions de l'image
            dimension_image = get_image_dimensions(image_path)

            # Déterminer le `Content-Type`
            content_type = (
                "image/jpeg"
                if image_path.lower().endswith((".jpg", ".jpeg"))
                else "image/png"
            )
            headers["Content-Type"] = content_type

            # Envoie l'image à l'API
            response = requests.post(API_URL, headers=headers, data=image_data)

            if response.status_code != 200:
                print(
                    f"Erreur lors de l'appel à l'API pour {image_path} : {response.status_code} - {response.text}"
                )
                batch_segmentations.append(None)
                continue

            # Traite la réponse de l'API
            results = response.json()

            # Crée les masques à partir des résultats
            mask = create_masks(results, dimension_image[0], dimension_image[1])

            if mask is None or mask.size == 0:
                print(
                    f"Le masque de segmentation pour {image_path} est vide ou n'a pas été créé correctement."
                )
                batch_segmentations.append(None)
                continue

            batch_segmentations.append(mask)
            time.sleep(
                2
            )  # Pause de 2 secondes entre les appels API pour éviter de surcharger le serveur

        except Exception as e:
            print(f"Une erreur est survenue pour {image_path} : {e}")
            batch_segmentations.append(None)

    return batch_segmentations


# Appeler la fonction pour segmenter les images listées dans image_paths
if image_paths:
    print(f"\nTraitement de {len(image_paths)} image(s) en batch...")
    batch_seg_results = segment_images_batch(image_paths)
    print("Traitement en batch terminé.")
else:
    batch_seg_results = []
    print("Aucune image à traiter en batch.")


def display_segmented_images_batch(original_image_paths, segmentation_masks):
    """
    Affiche les images originales et leurs masques segmentés.

    Args:
        original_image_paths (list): Liste des chemins des images originales.
        segmentation_masks (list): Liste des masques segmentés (NumPy arrays).
    """
    for i, (image_path, mask) in enumerate(
        zip(original_image_paths, segmentation_masks)
    ):
        if mask is None:
            print(f"Pas de masque pour l'image {image_path}.")
            continue

        original_image = Image.open(image_path)
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


# Afficher les résultats du batch
print("\nAffichage des résultats de la segmentation en batch...")
if batch_seg_results:
    display_segmented_images_batch(image_paths, batch_seg_results)
else:
    print("Aucun résultat de segmentation à afficher.")
