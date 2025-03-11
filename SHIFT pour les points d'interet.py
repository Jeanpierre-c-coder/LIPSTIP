import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt

def build_image_path(relative_path, base_path):
    """
    Construit le chemin complet de l'image.
    Si le chemin relatif commence par le nom du dossier de base (ex: 'logos_dataset'),
    il est retiré pour éviter une duplication.
    """
    # Récupérer le dernier dossier de base_path
    base_last = os.path.basename(base_path)
    # Normaliser la casse pour la comparaison
    if relative_path.lower().startswith(base_last.lower()):
        # Retirer la partie du dossier du chemin relatif
        relative_path = relative_path[len(base_last):]
        # Enlever d'éventuels séparateurs en début de chaîne
        relative_path = relative_path.lstrip("/\\")
    return os.path.join(base_path, relative_path)

def extract_sift_features(image_path):
    """
    Charge l'image en niveaux de gris et extrait les caractéristiques SIFT.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"L'image à {image_path} n'a pas pu être chargée.")
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return keypoints, descriptors

def match_sift_features(desc1, desc2):
    """
    Effectue le matching des descripteurs SIFT entre deux images.
    """
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def visualize_matches(image_path1, image_path2):
    """
    Visualise les correspondances SIFT entre deux images.
    """
    # Chargement des images en niveaux de gris
    img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        raise ValueError("Une ou plusieurs images n'ont pas pu être chargées.")
    
    # Extraction des points d'intérêt et des descripteurs SIFT
    kp1, desc1 = extract_sift_features(image_path1)
    kp2, desc2 = extract_sift_features(image_path2)
    
    # Matching des descripteurs SIFT
    matches = match_sift_features(desc1, desc2)
    
    # Visualisation des 10 meilleures correspondances
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)
    plt.figure(figsize=(12, 6))
    plt.imshow(img_matches, cmap='gray')
    plt.axis('off')
    plt.show()

# Définir le chemin de base où se trouvent vos images
base_path = r"C:\Users\HP\Desktop\LIPSTIP\Logos_dataset" 

# Chargement du DataFrame depuis un fichier CSV (modifiez le chemin vers votre fichier CSV)
csv_path = r"C:\Users\HP\Desktop\LIPSTIP\extracted_paths_final_LIPSTIP.csv"
df = pd.read_csv(csv_path)

# Vérification que les colonnes nécessaires existent dans le DataFrame
if 'main_mark_image' not in df.columns or 'earlier_mark_image' not in df.columns:
    raise KeyError("Le DataFrame doit contenir les colonnes 'main_mark_image' et 'earlier_mark_image'.")

# Construction des chemins complets pour la première paire d'images
image_main = build_image_path(df['main_mark_image'][0], base_path)
image_earlier = build_image_path(df['earlier_mark_image'][0], base_path)

# Vérification de l'existence des fichiers
if not os.path.exists(image_main):
    raise FileNotFoundError(f"Le fichier n'existe pas : {image_main}")
if not os.path.exists(image_earlier):
    raise FileNotFoundError(f"Le fichier n'existe pas : {image_earlier}")

# Appel de la fonction visualize_matches
visualize_matches(image_main, image_earlier)
