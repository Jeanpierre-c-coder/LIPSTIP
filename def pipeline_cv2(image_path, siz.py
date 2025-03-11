import cv2
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch  # Si besoin de manipuler les tensors

def preprocess_image_cv2(image_path, size=(224, 224)):
    """
    Charge et pré-traite l'image avec OpenCV.
    L'image est chargée en couleur (BGR), convertie en RGB,
    redimensionnée et normalisée (mise à l'échelle entre 0 et 1).

    Args:
        image_path (str): Chemin vers l'image.
        size (tuple): Taille souhaitée de l'image (largeur, hauteur).

    Returns:
        numpy.ndarray: Image prétraitée.
    """
    # Chargement en couleur (BGR)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"L'image à {image_path} n'a pas pu être chargée.")
    # Conversion BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Redimensionnement
    img = cv2.resize(img, size)
    # Normalisation : mise à l'échelle entre 0 et 1
    img = img.astype('float32') / 255.0
    return img

# Définition du pipeline de transformation avec Torchvision (pour PIL)
transform_pipeline = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),  # Convertit l'image en tensor et met à l'échelle [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def preprocess_image_pil(image_path):
    """
    Charge et pré-traite l'image en utilisant PIL et le pipeline Torchvision.
    L'image est convertie en format PIL, puis transformée en tensor.

    Args:
        image_path (str): Chemin vers l'image.

    Returns:
        torch.Tensor: Image prétraitée sous forme de tensor.
    """
    img = Image.open(image_path).convert('RGB')
    return transform_pipeline(img)

# Exemple d'utilisation
if __name__ == "__main__":
    # Correction du chemin d'accès en utilisant une chaîne brute
    image_path = r"C:\Users\HP\Desktop\LIPSTIP\Logos_dataset\earlier_005271598.jpg"
    
    # Vérifiez que le fichier existe à cet emplacement.
    # Vous pouvez également utiliser os.path.exists pour vérifier avant de charger l'image.
    import os
    if not os.path.exists(image_path):
        print(f"Le fichier n'existe pas : {image_path}")
    else:
        # Prétraitement avec OpenCV
        try:
            processed_img_cv2 = preprocess_image_cv2(image_path)
            plt.figure(figsize=(6, 6))
            plt.imshow(processed_img_cv2)
            plt.title("Prétraitement avec OpenCV")
            plt.axis('off')
            plt.show()
        except Exception as e:
            print("Erreur lors du prétraitement avec OpenCV :", e)

        # Prétraitement avec PIL et Torchvision
        try:
            processed_img_pil = preprocess_image_pil(image_path)
            # Pour afficher le tensor, on réarrange les dimensions (C, H, W) -> (H, W, C)
            img_np = processed_img_pil.permute(1, 2, 0).numpy()
            # Les valeurs peuvent être négatives ou dépasser 1 en raison de la normalisation.
            # On peut les retransformer pour l'affichage (optionnel).
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
            plt.figure(figsize=(6, 6))
            plt.imshow(img_np)
            plt.title("Prétraitement avec PIL & Torchvision")
            plt.axis('off')
            plt.show()
        except Exception as e:
            print("Erreur lors du prétraitement avec PIL & Torchvision :", e)
