import pandas as pd  # Importation de pandas
import os  # Pour vérifier les chemins
from PIL import Image  # Pour manipuler les images
import matplotlib.pyplot as plt  # Pour afficher les images

# Définir le chemin de travail
base_path = r'c:/Users/HP/Desktop/LIPSTIP'

# Charger le fichier CSV en utilisant le chemin absolu
csv_path = os.path.join(base_path, 'extracted_paths_final_LIPSTIP.csv')
df = pd.read_csv(csv_path)

# Afficher un aperçu des données
print(df.head())

# Vérifier l'existence des fichiers d'images
def check_image_paths(df):
    for index, row in df.iterrows():
        main_image_path = os.path.join(base_path, row['main_mark_image'])
        earlier_image_path = os.path.join(base_path, row['earlier_mark_image'])
        if not (os.path.exists(main_image_path) and os.path.exists(earlier_image_path)):
            print(f"Chemin invalide à la ligne {index} : {main_image_path}, {earlier_image_path}")

check_image_paths(df)

# Fonction pour visualiser une paire d'images
def visualize_pair(main_image_path, earlier_image_path):
    try:
        main_image = Image.open(main_image_path)
        earlier_image = Image.open(earlier_image_path)

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(main_image)
        plt.axis('off')
        plt.title('Main Mark')

        plt.subplot(1, 2, 2)
        plt.imshow(earlier_image)
        plt.axis('off')
        plt.title('Earlier Mark')

        plt.show()
    except FileNotFoundError:
        print(f"Fichier introuvable : {main_image_path} ou {earlier_image_path}")

# Visualiser la première paire d'images si les chemins existent
main_image_first = os.path.join(base_path, df['main_mark_image'][0])
earlier_image_first = os.path.join(base_path, df['earlier_mark_image'][0])
if os.path.exists(main_image_first) and os.path.exists(earlier_image_first):
    visualize_pair(main_image_first, earlier_image_first)
else:
    print("Les fichiers de la première ligne sont introuvables.")
