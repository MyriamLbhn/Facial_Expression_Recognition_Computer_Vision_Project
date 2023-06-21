import streamlit as st
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
import moviepy as mpy

# Charger le modèle
model = YOLO("best_model.pt")

# Titre de l'application Streamlit
st.title("Reconnaissance faciale d'émotion avec YOLO")



# Ajouter un menu déroulant dans la sidebar
option = st.sidebar.selectbox("Sélectionner le type de média", ("Image", "Vidéo"))

if option == "Image":

    # Sélectionner une image à partir du fichier local
    uploaded_file = st.file_uploader("Choisir une image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Charger l'image
        image = Image.open(uploaded_file)

        # Afficher l'image originale à gauche
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption='Image originale', use_column_width=True)

        # Vérifier et ajuster le format de l'image
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Bouton de prédiction
        if st.button("Prédire"):
            # Convertir l'image en tableau numpy
            image_np = np.array(image)

            # Prédire sur l'image chargée
            res = model.predict(image_np)

            # Afficher le résultat de détection à droite
            with col2:
                res_plotted = res[0].plot()
                st.image(res_plotted, caption='Résultat de détection', use_column_width=True)
