import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
model_path = "pokemon-model_transferlearning.keras"
model = tf.keras.models.load_model(model_path)

# Define the core prediction function
def predict_pokemon(image):
    # Preprocess image
    image = image.resize((150, 150))  # Resize the image to 150x150
    image = np.array(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    
    # Predict
    prediction = model.predict(image)
    
    # Apply softmax to get probabilities for each class
    probabilities = tf.nn.softmax(prediction, axis=1)
    
    # Map probabilities to Pokemon classes
    pokemon_classes = ['Articuno', 'Bulbasaur', 'Charmander']  
    probabilities_dict = {pokemon_class: round(float(probability), 2) for pokemon_class, probability in zip(pokemon_classes, probabilities.numpy()[0])}
    
    return probabilities_dict

# Streamlit interface
st.title("Pokemon Classifier")
st.write("A simple MLP classification model for image classification using a pretrained model.")

# Upload image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    
    predictions = predict_pokemon(image)
    
    st.write(predictions)

# Example images
st.sidebar.title("Examples")
example_images = ["images/01.jpg", "images/02.png", "images/03.png", "images/04.jpg", "images/05.png", "images/06.png"]
for example_image in example_images:
    st.sidebar.image(example_image, use_column_width=True)
