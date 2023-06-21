import streamlit as st
import torch
import cv2
import numpy as np
from ultralytics import YOLO

import streamlit as st
from PIL import Image
import cv2

# Charger le modèle
model = YOLO("best_model.pt")

# Titre de l'application Streamlit
st.title("Détection d'objets avec YOLO")

# Sélectionner une image à partir du fichier local
uploaded_file = st.file_uploader("Choisir une image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Charger l'image
    image = Image.open(uploaded_file)
    st.image(image, caption='Image originale', use_column_width=True)

    # Prédire sur l'image chargée
    res = model.predict(image)

    # Afficher le résultat
    res_plotted = res[0].plot()
    st.image(res_plotted, caption='Résultat de détection', use_column_width=True)
