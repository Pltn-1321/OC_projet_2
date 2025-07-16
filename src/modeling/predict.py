import requests
import time
from features import get_image_dimensions, create_masks  # parent directory


def segment_image(image_path, api_url, api_token):
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


def segment_images_batch(image_paths, api_url, api_token, sleep=2):
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
