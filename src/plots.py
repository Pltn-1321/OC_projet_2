import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from typing import List, Optional, Dict
from features import CLASS_MAPPING, colorize_mask
import matplotlib.patches as mpatches  # Ajout de l'import
import cv2


def display_segmented_images_batch(
    original_image_paths: List[str], segmentation_masks: List[Optional[np.ndarray]]
) -> None:
    """
    Affiche côte à côte chaque image originale et son masque de segmentation.

    Args:
        original_image_paths (List[str]): Liste des chemins d'accès des images originales.
        segmentation_masks (List[Optional[np.ndarray]]): Liste des masques segmentés
                                                        (arrays 2D ou None si échec).

    Returns:
        None: Affiche les paires image originale / masque dans des fenêtres matplotlib.
    """

    if len(original_image_paths) != len(segmentation_masks):
        print(
            f"Erreur : Nombre d'images ({len(original_image_paths)}) différent du nombre de masques ({len(segmentation_masks)})"
        )
        return

    for i, (image_path, mask) in enumerate(
        zip(original_image_paths, segmentation_masks)
    ):
        # Vérifications simples
        if mask is None:
            print(f"[{i + 1}] Pas de masque pour l'image {image_path}")
            continue

        if not isinstance(mask, np.ndarray) or len(mask.shape) != 2:
            print(f"[{i + 1}] Masque invalide pour l'image {image_path}")
            continue

        # Affichage
        try:
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

        except Exception as e:
            print(f"[{i + 1}] Impossible d'afficher l'image {image_path}: {str(e)}")


def display_triplet_with_scores(
    img_path: str,
    mask_true: np.ndarray,
    mask_pred: np.ndarray,
    iou_mean: float,
    iou_per_class: Dict[int, float],
    class_names: Optional[Dict[int, str]] = None,
    colormap="jet",
) -> None:
    """
    Affiche côte à côte l'image originale, le masque de vérité terrain et le masque prédit, avec IoU.
    Args:
        img_path (str): Chemin vers l'image originale.
        mask_true (np.ndarray): Masque de vérité terrain (H, W).
        mask_pred (np.ndarray): Masque prédit (H, W).
        iou_mean (float): Score IoU moyen.
        iou_per_class (dict): Détail IoU par classe.
        class_names (dict): Optionnel. Mapping id classe -> nom.
        colormap (str): Colormap matplotlib pour affichage des masques.
    """
    img = Image.open(img_path)

    plt.figure(figsize=(16, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Image originale")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask_true, cmap=colormap)
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(mask_pred, cmap=colormap)
    plt.title("Masque prédit")
    plt.axis("off")

    # Affiche le score moyen et par classe en légende
    class_legend = ""
    for cid, score in iou_per_class.items():
        cname = (
            class_names[cid] if class_names and cid in class_names else f"Classe {cid}"
        )
        class_legend += f"{cname}: {score:.2f}\n"

    plt.suptitle(
        f"IoU moyen: {iou_mean:.2f}\n{class_legend}",
        fontsize=14,
        y=0.98,
        ha="center",
        color="navy",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()


def display_triplet_with_scores_2(
    img_path: str,
    mask_true: np.ndarray,
    mask_pred: np.ndarray,
    iou_mean: float,
    iou_per_class: Dict[int, float],
    class_names: Dict[int, str],
    colormap: Dict[int, tuple],
    max_classes: int = 9,
) -> None:
    """
    Affiche image originale, ground truth, prédiction, IoU + légende couleurs.
    Args:
        img_path (str): Chemin image.
        mask_true (np.ndarray): GT.
        mask_pred (np.ndarray): préd.
        iou_mean (float): IoU moyen.
        iou_per_class (dict): IoU par classe.
        class_names (dict): mapping id->nom.
        colormap (dict): mapping id->(R,G,B).
        max_classes (int): max classes in legend.
    """
    import matplotlib.pyplot as plt
    from PIL import Image
    import numpy as np

    img = Image.open(img_path)

    # Colorisation
    mask_true_col = colorize_mask(mask_true, colormap)
    mask_pred_col = colorize_mask(mask_pred, colormap)

    fig, axes = plt.subplots(1, 3, figsize=(14, 6))
    axes[0].imshow(img)
    axes[0].set_title("Image originale")
    axes[0].axis("off")
    axes[1].imshow(mask_true_col)
    axes[1].set_title("Vérité terrain")
    axes[1].axis("off")
    axes[2].imshow(mask_pred_col)
    axes[2].set_title("Masque prédit")
    axes[2].axis("off")

    # Titre général
    fig.suptitle(f"IoU moyen : {iou_mean:.2f}", fontsize=16, color="navy")

    # Légende couleurs/classes IoU, sur le côté
    legend_elements = []
    for cid, cname in class_names.items():
        color = tuple(np.array(colormap.get(cid, (0, 0, 0))) / 255)
        score = iou_per_class.get(cid, None)
        score_str = f"{score:.2f}" if score is not None and not np.isnan(score) else "—"
        patch = mpatches.Patch(color=color, label=f"{cname}: {score_str}")
        legend_elements.append(patch)

    # On place la légende à droite, en dehors du cadre image
    plt.legend(
        handles=legend_elements[
            :max_classes
        ],  # pour ne pas tout mettre si bcp de classes
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0.0,
        title="Classes (IoU)",
    )
    plt.tight_layout(rect=[0, 0, 0.85, 0.95])
    plt.show()


def display_segmentation_viz(
    img_path: str,
    mask: np.ndarray,
    colormap: dict,
    class_names: dict,
    alpha_overlay: float = 0.6,
):
    """
    Affiche image originale, masque segmenté colorisé avec légende incrustée, overlay masque + image.
    - img_path : chemin image originale
    - mask : mask 2D (id de classe par pixel)
    - colormap : {class_id: (R,G,B)}
    - class_names : {class_id: "Nom classe"}
    - alpha_overlay : transparence overlay
    """
    # Ouverture de l'image
    img = np.array(Image.open(img_path).convert("RGB"))
    mask_color = colorize_mask(mask, colormap)

    # Overlay (img * (1-alpha) + mask * alpha)
    mask_overlay = (img * (1 - alpha_overlay) + mask_color * alpha_overlay).astype(
        np.uint8
    )

    # Setup figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 7))
    axes[0].imshow(img)
    axes[0].set_title("Image originale")
    axes[0].axis("off")

    # ----- Affichage masque avec légende incrustée -----
    axes[1].imshow(mask_color)
    axes[1].set_title("Masque segmenté")
    axes[1].axis("off")

    # On incruste la légende sur l'image de gauche
    y0 = 15
    dy = 20
    x0 = 10
    for idx, class_id in enumerate(sorted(class_names)):
        cname = class_names[class_id]
        color = tuple(np.array(colormap[class_id]) / 255)
        axes[1].text(
            x0,
            y0 + idx * dy,
            f"{cname}",
            color=color,
            fontsize=12,
            weight="bold",
            va="top",
            ha="left",
            bbox=dict(facecolor="black", alpha=0.8, boxstyle="round,pad=0.2"),
        )
        # On peut ajouter une pastille couleur à côté
        axes[1].add_patch(
            plt.Rectangle(
                (x0 - 18, y0 - 4 + idx * dy), 14, 14, color=color, clip_on=False
            )
        )

    # Overlay image + masque
    axes[2].imshow(mask_overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


def show_image_triplet_with_legend_iou_per_class(
    img_path: str,
    mask_true: np.ndarray,
    mask_pred: np.ndarray,
    custom_colormap: dict,
    legend_labels: dict,
    iou_mean: float,
    iou_per_class: dict,
    overlay_alpha: float = 0.6,
):
    """
    Affiche côte à côte l'image originale, le masque terrain, le masque prédit (en overlay),
    et une légende dynamique avec l'IoU par classe.

    Args:
        img_path (str): Chemin de l'image source.
        mask_true (np.ndarray): Masque de vérité terrain (2D, valeurs de classes).
        mask_pred (np.ndarray): Masque prédit (2D, valeurs de classes).
        custom_colormap (dict): Dictionnaire {class_id: (R, G, B)} pour coloriser les masques.
        legend_labels (dict): Dictionnaire {class_id: nom de la classe} ou {str(class_id): nom}.
        iou_mean (float): Score IoU moyen pour l'image.
        iou_per_class (dict): Dictionnaire {class_id: score IoU pour chaque classe}.
        overlay_alpha (float): Opacité du masque prédit en overlay (0 = invisible, 1 = opaque).

    Returns:
        None: Affiche la figure Matplotlib.

    Astuces :
        - Modifie "gridspec_kw" pour changer la taille relative de la légende.
        - Tu peux facilement adapter pour sauvegarder la figure (plt.savefig).
    """
    # Chargement et conversion de l'image
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    mask_true_color = colorize_mask(mask_true, custom_colormap)
    mask_pred_color = colorize_mask(mask_pred, custom_colormap)

    # Overlay du masque prédit sur l'image originale
    overlay_pred = cv2.addWeighted(
        img, 1 - overlay_alpha, mask_pred_color, overlay_alpha, 0
    )

    # Préparation du plot
    fig, axs = plt.subplots(
        1, 4, figsize=(15, 5), gridspec_kw={"width_ratios": [3, 3, 3, 2]}
    )

    axs[0].imshow(img)
    axs[0].set_title("Image originale")
    axs[0].axis("off")

    axs[1].imshow(mask_true_color)
    axs[1].set_title("Masque terrain")
    axs[1].axis("off")

    axs[2].imshow(overlay_pred)
    axs[2].set_title(f"Mask préd. (IoU moyen={iou_mean:.2f})")
    axs[2].axis("off")

    # Préparer la légende dynamique
    handles = []
    # Tri des classes par IoU décroissant (classes absentes tout en bas)
    items = sorted(
        iou_per_class.items(), key=lambda x: -x[1] if not np.isnan(x[1]) else -999
    )
    for class_id, score in items:
        if np.isnan(score):
            continue  # On n'affiche pas les classes absentes de l'image
        # Gestion clé int ou str
        label = legend_labels.get(
            class_id, legend_labels.get(str(class_id), str(class_id))
        )
        color = np.array(custom_colormap.get(int(class_id), (0, 0, 0))) / 255
        handles.append(
            mpatches.Patch(
                color=color,
                label=f"{label} (IoU: {score:.2f})",
            )
        )
    axs[3].legend(
        handles=handles, loc="center left", fontsize=10, frameon=False, ncol=1
    )
    axs[3].axis("off")

    plt.tight_layout()
    plt.show()
