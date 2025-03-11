import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# Définir le chemin de base où se trouvent les images
base_path = "chemin/vers/les/images"  # Remplacez par le chemin réel vers vos images

# Charger le DataFrame depuis un fichier CSV
# Le fichier CSV doit contenir les colonnes 'main_mark_image' et 'earlier_mark_image'
csv_path = "chemin/vers/le/fichier/images.csv"  # Remplacez par le chemin réel vers votre fichier CSV
df = pd.read_csv(csv_path)

def visualize_pair(main_image_path, earlier_image_path):
    try:
        main_img = Image.open(main_image_path)
        earlier_img = Image.open(earlier_image_path)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(main_img)
        plt.axis('off')
        plt.title('Main Mark')
        plt.subplot(1, 2, 2)
        plt.imshow(earlier_img)
        plt.axis('off')
        plt.title('Earlier Mark')
        plt.show()
    except Exception as e:
        print(e)

# Exemple pour la première paire d'images
visualize_pair(os.path.join(base_path, df['main_mark_image'][0]),
               os.path.join(base_path, df['earlier_mark_image'][0]))
