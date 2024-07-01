import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
import os

# Charger le modèle sauvegardé
model_opt_loaded = tf.keras.models.load_model('model_final_notebook.h5')

# Fonction pour faire des prédictions
def predict(image):
    image = image.resize((224, 224))  # Adapter la taille de l'image
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normaliser l'image
    predictions = model_opt_loaded.predict(image)
    return predictions

# Interface utilisateur avec Streamlit
st.title('Détection de la race de chien')
st.write('Téléchargez une image de chien et le modèle prédit sa race.')

# Téléchargement de l'image par l'utilisateur
uploaded_file = st.file_uploader("Choisissez une image de chien...", type="jpg")

# Chemin vers le dossier contenant les images de chiens
dogImages_folder = 'data/SpecificRaces'

# Récupérer les noms des sous-dossiers
classes = os.listdir(dogImages_folder)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Image téléchargée', use_column_width=True)
    st.write("Classification...")
    predictions = predict(image)
    # Afficher les résultats
    predicted_class = classes[np.argmax(predictions)]
    st.write(f'La race prédite est : {predicted_class} avec une probabilité de {np.max(predictions)*100:.2f}%')
