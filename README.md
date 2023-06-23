# Facial_Expression_Recognition_Computer_Vision_Project

## Objectifs

L'objectif du projet est concevoir une application qui intègre YOLO-v8, le dernier modèle d'IA pour la vision par ordinateur en 2023.
Le modèle YOLOv8n a été entrainé sur un dataset pour permettre la reconnaissance des émotions du visage (anger, fear, happy, sad, neutral).    
Le dataset utilisé pour l'entraînement est disponible sur Roboflow, [ici](https://universe.roboflow.com/emotiondetection/facedetection2-6qc02)


## Arborescence des fichiers

Ce projet est organisé en plusieurs fichiers :

*Model_YOLO_Facial_Recognition.ipynb* : notebook pour l'entraînement du modèle yolo avec le dataset roboflow.

*best_model.pt* : modèle yolov8n entrainé sur le dataset de reconnaissance d'émotion.  

*utils.py* : fonctions nescessaire au fonctionnemment de l'application Streamlit.

*app.py* : script de l'application


## Fonctionnement de l'application

L'application de reconnaissance faciale d'émotion propose 3 options à choisir dans le menu latéral :
- image : permet d'upload une image puis un clic sur le bouton prédire affiche la prédiction à côté de l'image originale
- vidéo : permet d'upload une vidéo puis la vidéo avec les résultats de la prédiction s'affiche
- webcam : permet d'obtenir des prédictions en live en se connectant à la webcam de l'utilisateur

## Requirements

Installez les bibliothèques requises à l'aide de `pip install -r requirements.txt` ou installez-les manuellement.


## Lancer l'application Streamlit

Exécutez la commande suivante pour lancer l'application : `streamlit run app.py`  
Streamlit ouvrira automatiquement un navigateur et affichera l'application. 