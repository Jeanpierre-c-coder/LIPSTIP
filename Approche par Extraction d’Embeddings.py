# IMPORTANT :
# Pour exécuter ce script, assurez-vous d'utiliser des guillemets autour du chemin complet,
# car il contient des espaces.
# PowerShell :
# & "C:\Users\HP\AppData\Local\Microsoft\WindowsApps\python3.13.exe" "C:\Users\HP\Desktop\LIPSTIP\Approche par Extraction dEmbeddings.py"

import torch
from torchvision import models, transforms
from PIL import Image

# Définir le device global (GPU si disponible, sinon CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Chargement du modèle pré-entraîné
# (Note : dans les versions récentes de PyTorch, vous pouvez utiliser weights=models.ResNet50_Weights.DEFAULT)
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
    Extrait l'embedding d'une image en utilisant un modèle ResNet50 pré-entraîné.
    
    Args:
        image_path (str): Chemin vers l'image.
    
    Returns:
        torch.Tensor: Les features extraites sous forme d'un vecteur plat.
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

# Exemple d'utilisation
if __name__ == "__main__":
    image_path = r"C:\Users\HP\Desktop\LIPSTIP\Logos_dataset\earlier_003466547.jpg"  # Remplacez par le chemin réel de votre image
    embedding = get_embedding(image_path)
    print("Embedding shape:", embedding.shape)
