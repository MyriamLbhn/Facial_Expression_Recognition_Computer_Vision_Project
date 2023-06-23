import streamlit as st
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
import moviepy as mpy

# Charger le modèle
model = YOLO("best_model.pt")

# Titre de l'application Streamlit
st.title("Reconnaissance faciale d'émotion")



# Ajouter un menu déroulant dans la sidebar
option = st.sidebar.selectbox("Sélectionner le type de média", ("Image", "Vidéo", "Webcam"))

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

elif option == "Vidéo":
    # Sélectionner une vidéo à partir du fichier local
    uploaded_file = st.file_uploader("Choisir une vidéo", type=["mp4", "mov"])

    if uploaded_file is not None:
        # Enregistrer le fichier téléchargé sur le disque
        with open('uploaded_video.mp4', 'wb') as f:
            f.write(uploaded_file.getbuffer())

        # Charger la vidéo
        video = mpy.VideoFileClip('uploaded_video.mp4')
        
        # reduire les fps de la vidéo
        video = video.set_fps(30)
        # rediot ma taille de la vidéo  à 480p
        video = video.resize(height=480)

        # Créer une fonction pour traiter chaque image de la vidéo
        def process_image(image):
            # Convertir l'image en tableau numpy
            image_np = np.array(image)

            # Prédire sur l'image
            res = model.predict(image_np)

            # Dessiner les détections sur l'image
            res_plotted = res[0].plot()

            # Convertir l'image de retour en format PIL
            image_pil = Image.fromarray(res_plotted)

            return np.array(image_pil)

        # Indiquer que le traitement est en cours
        with st.spinner('Processing the video...'):
            # Appliquer la fonction de traitement à chaque image de la vidéo
            processed_video = video.fl_image(process_image)

            # Enregistrer la vidéo traitée
            processed_video.write_videofile("processed_video.mp4", codec='libx264')

        # Afficher la vidéo traitée
        st.video("processed_video.mp4")

elif option == "Webcam":
     # Capture vidéo à partir de la webcam
    video_capture = cv2.VideoCapture(0)

    # Créer un espace pour afficher l'image annotée
    image_placeholder = st.empty()

    # Boucle principale de l'application Streamlit
    while True:
        # Lire la vidéo frame par frame
        ret, frame = video_capture.read()

        # Redimensionner la frame si nécessaire
        if frame.shape[1] > 800:
            frame = cv2.resize(frame, (800, int(frame.shape[0] * 800 / frame.shape[1])))

        # Convertir l'image en format approprié
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Utiliser le modèle YOLO pour prédire les détections
        res = model.predict(img)

        # Dessiner les détections sur l'image
        res_plotted = res[0].plot()

        # Convertir l'image annotée en format BGR pour l'affichage avec OpenCV
        annotated_frame = cv2.cvtColor(res_plotted, cv2.COLOR_RGB2BGR)

        # Afficher l'image annotée dans Streamlit
        image_placeholder.image(annotated_frame, channels="BGR", caption='Résultat de détection', use_column_width=True)

        # Attendre que l'utilisateur appuie sur une touche pour quitter l'application
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libérer les ressources
    video_capture.release()
    cv2.destroyAllWindows()
