import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import random
import torch
from torchvision import models, transforms

# --- Fonction pour construire le chemin de l'image ---
def build_image_path(base_path, relative_path):
    """
    Construit le chemin complet vers l'image en supprimant une éventuelle duplication
    du dossier de base (ex. 'Logos_dataset') si présent dans le chemin relatif.
    """
    base_folder = os.path.basename(os.path.normpath(base_path))
    if relative_path.startswith(base_folder):
        relative_path = relative_path[len(base_folder):].lstrip("/\\")
    return os.path.join(base_path, relative_path)

# --- Configuration du modèle et du prétraitement ---

# Définir le device global 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Chargement du modèle pré-entraîné (ResNet50)
model = models.resnet50(pretrained=True)
model.eval()  # Passage en mode évaluation
model.to(device)  # Déplacer le modèle sur le device

# On retire la dernière couche fully connected pour obtenir les embeddings
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
feature_extractor.to(device)  # Déplacer l'extracteur sur le device

# Définition du pipeline de prétraitement de l'image
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),  # Conversion en tensor avec valeurs dans [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def get_embedding(image_path):
    """
    Extrait l'embedding d'une image en utilisant le modèle ResNet50 pré-entraîné.
    
    Args:
        image_path (str): Chemin vers l'image.
    
    Returns:
        torch.Tensor: Vecteur d'embedding aplati.
    """
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        raise ValueError(f"Erreur lors de l'ouverture de l'image {image_path} : {e}")
    
    # Appliquer le prétraitement
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0)  # Ajoute la dimension batch
    
    # Déplacer le batch sur le device approprié
    input_batch = input_batch.to(device)
    
    with torch.no_grad():
        features = feature_extractor(input_batch)
    
    # Aplatir le résultat pour obtenir un vecteur 1D par image
    features = features.view(features.size(0), -1)
    return features

# --- Chargement des données et extraction des embeddings ---

# Chemin vers le fichier CSV contenant la colonne 'main_mark_image'
csv_path = r"C:\Users\HP\Desktop\LIPSTIP\extracted_paths_final_LIPSTIP.csv"  # Mettez à jour ce chemin si nécessaire

# Charger le DataFrame
df = pd.read_csv(csv_path)

# Chemin de base où se trouvent les images
base_path = r"C:\Users\HP\Desktop\LIPSTIP\Logos_dataset"

# Extraction des embeddings pour chaque image
embeddings = {}
for idx, row in df.iterrows():
    # Utilisation de build_image_path pour éviter la duplication du dossier
    main_img_path = build_image_path(base_path, row['main_mark_image'])
    emb = get_embedding(main_img_path).cpu().numpy().flatten()  # S'assurer que l'embedding est sur CPU
    embeddings[row['main_mark_image']] = emb

# --- Sélection d'images aléatoires pour l'exploration ---
selected_images = random.sample(list(embeddings.keys()), 3)

# Affichage en console des 5 images les plus proches pour chaque image sélectionnée
for img in selected_images:
    distances = {}
    for other_img, emb in embeddings.items():
        if other_img == img:
            continue
        dist = np.linalg.norm(embeddings[img] - emb)
        distances[other_img] = dist
    similar_images = sorted(distances, key=distances.get)[:5]
    print(f"\nPour l'image {img}, les 5 plus proches sont :")
    for sim_img in similar_images:
        print(sim_img)

# --- Fonction d'affichage des images similaires ---
def display_similar_images(reference_image, similar_images):
    plt.figure(figsize=(15, 5))
    # Affichage de l'image de référence
    plt.subplot(1, len(similar_images) + 1, 1)
    ref_img = Image.open(build_image_path(base_path, reference_image))
    plt.imshow(ref_img)
    plt.title("Référence")
    plt.axis('off')
    # Affichage des images similaires
    for i, sim in enumerate(similar_images):
        plt.subplot(1, len(similar_images) + 1, i + 2)
        sim_img = Image.open(build_image_path(base_path, sim))
        plt.imshow(sim_img)
        plt.title(f"Sim {i+1}")
        plt.axis('off')
    plt.show()

# --- Affichage graphique des images similaires pour chaque image sélectionnée ---
for img in selected_images:
    distances = {}
    for other_img, emb in embeddings.items():
        if other_img == img:
            continue
        dist = np.linalg.norm(embeddings[img] - emb)
        distances[other_img] = dist
    similar_images = sorted(distances, key=distances.get)[:5]
    display_similar_images(img, similar_images)
