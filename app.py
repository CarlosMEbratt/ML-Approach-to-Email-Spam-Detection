import streamlit as st
import requests
import joblib
import os
import tempfile
from sklearn.feature_extraction.text import TfidfVectorizer

# Predefined GitHub URL for the model file
GITHUB_MODEL_URL = "https://raw.githubusercontent.com/CarlosMEbratt/ML-Approach-to-Email-Spam-Detection/main/rf_model.pkl"  # Replace with your actual GitHub URL

# Function to load a model from a GitHub repository
def load_model_from_github(url):
    response = requests.get(url)
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(response.content)
        model = joblib.load(tmp_file.name)
    return model

# Function to load a model from an uploaded file
def load_model_from_upload(uploaded_file):
    return joblib.load(uploaded_file)

# Function to predict spam or ham
def predict_message(model, text):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform([text])
    prediction = model.predict(X)
    return "Spam" if prediction == 1 else "Ham"

# Streamlit app
st.title("Spam/Ham Classifier")

# Option to select the model source
model_option = st.radio(
    "Choose a model source:",
    ("Load from GitHub", "Upload your own model")
)

model = None

# Load the model based on user selection
if model_option == "Load from GitHub":
    if st.button("Load Model"):
        try:
            model = load_model_from_github(GITHUB_MODEL_URL)
            st.success("Model loaded successfully from GitHub!")
        except Exception as e:
            st.error(f"Failed to load model: {e}")
elif model_option == "Upload your own model":
    uploaded_file = st.file_uploader("Upload your model file (.pkl):", type=["pkl"])
    if uploaded_file:
        try:
            model = load_model_from_upload(uploaded_file)
            st.success("Model uploaded and loaded successfully!")
        except Exception as e:
            st.error(f"Failed to load model: {e}")

# Field to enter text for prediction
text_input = st.text_area("Enter the text message for classification:")

# Predict button
if st.button("Predict"):
    if model is not None and text_input.strip() != "":
        result = predict_message(model, text_input)
        st.write(f"The message is classified as: **{result}**")
    else:
        st.error("Please load a model and enter a message for classification.")
