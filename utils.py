# utils.py

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Charger le modèle
model = YOLO("best_model.pt")

def process_image(image):
    """
    Fonction pour traiter une image et effectuer une prédiction.
    Prend en entrée une image et renvoie l'image avec les détections dessinées.
    """
    # Convertir l'image en tableau numpy
    image_np = np.array(image)

    # Prédire sur l'image
    res = model.predict(image_np)

    # Dessiner les détections sur l'image
    res_plotted = res[0].plot()

    # Convertir l'image de retour en format PIL
    image_pil = Image.fromarray(res_plotted)

    return np.array(image_pil)

def process_video(video):
    """
    Fonction pour traiter une vidéo et effectuer une prédiction sur chaque image.
    Prend en entrée une vidéo et renvoie la vidéo avec les détections dessinées.
    """
    # Reduire les fps de la vidéo
    video = video.set_fps(30)

    # Reduire la taille de la vidéo à 480p
    video = video.resize(height=480)

    # Appliquer la fonction de traitement à chaque image de la vidéo
    processed_video = video.fl_image(process_image)

    return processed_video

def process_webcam(video_capture):
    """
    Fonction pour traiter une vidéo de webcam et effectuer une prédiction sur chaque image.
    Prend en entrée un objet de capture vidéo et renvoie un générateur d'images traitées.
    """
    while True:
        # Lire la vidéo frame par frame
        ret, frame = video_capture.read()

        # Redimensionner la frame si nécessaire
        if frame.shape[1] > 800:
            frame = cv2.resize(frame, (800, int(frame.shape[0] * 800 / frame.shape[1])))

        # Convertir l'image en format approprié
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Appliquer la fonction de traitement à chaque image
        processed_frame = process_image(img)

        yield processed_frame
