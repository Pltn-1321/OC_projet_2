import os


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
        raise FileNotFoundError(f"Le dossier {image_dir} n'existe pas.")
    if not os.path.isdir(image_dir):
        raise ValueError(f"{image_dir} n'est pas un répertoire.")

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
