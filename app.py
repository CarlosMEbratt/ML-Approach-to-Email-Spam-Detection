import streamlit as st
import pickle
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import hstack
import requests
from io import BytesIO

# NLTK Resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

GITHUB_MODEL_URL = "https://raw.githubusercontent.com/CarlosMEbratt/ML-Approach-to-Email-Spam-Detection/main/rf_model.pkl"
GITHUB_VECTOR_TFIDF_URL = "https://raw.githubusercontent.com/CarlosMEbratt/ML-Approach-to-Email-Spam-Detection/main/vectorizer_tfidf.pkl"
GITHUB_VECTOR_BIGRAMS_URL = "https://raw.githubusercontent.com/CarlosMEbratt/ML-Approach-to-Email-Spam-Detection/main/vectorizer_bigrams.pkl"
GITHUB_VECTOR_TRIGRAMS_URL = "https://raw.githubusercontent.com/CarlosMEbratt/ML-Approach-to-Email-Spam-Detection/main/vectorizer_trigrams.pkl"

def load_from_url(url):
    response = requests.get(url)
    return pickle.load(BytesIO(response.content))

def load_model():
    model = None
    vectorizer_tfidf = None
    vectorizer_bigrams = None
    vectorizer_trigrams = None
    
    option = st.sidebar.selectbox("Choose model source", ["Upload model file", "Load from GitHub"])
    
    if option == "Upload model file":
        uploaded_model = st.file_uploader("Choose a model file", type="pkl")
        uploaded_vector_tfidf = st.file_uploader("Choose a TF-IDF vectorizer file", type="pkl")
        uploaded_vector_bigrams = st.file_uploader("Choose a Bigrams vectorizer file", type="pkl")
        uploaded_vector_trigrams = st.file_uploader("Choose a Trigrams vectorizer file", type="pkl")
        
        if uploaded_model is not None and uploaded_vector_tfidf is not None and uploaded_vector_bigrams is not None and uploaded_vector_trigrams is not None:
            model = pickle.load(uploaded_model)
            vectorizer_tfidf = pickle.load(uploaded_vector_tfidf)
            vectorizer_bigrams = pickle.load(uploaded_vector_bigrams)
            vectorizer_trigrams = pickle.load(uploaded_vector_trigrams)
    elif option == "Load from GitHub":
        model = load_from_url(GITHUB_MODEL_URL)
        vectorizer_tfidf = load_from_url(GITHUB_VECTOR_TFIDF_URL)
        vectorizer_bigrams = load_from_url(GITHUB_VECTOR_BIGRAMS_URL)
        vectorizer_trigrams = load_from_url(GITHUB_VECTOR_TRIGRAMS_URL)
    
    if model is None or vectorizer_tfidf is None or vectorizer_bigrams is None or vectorizer_trigrams is None:
        st.sidebar.warning("Model or vectorizers not loaded. Please upload files or choose to load from GitHub.")
    
    return model, vectorizer_tfidf, vectorizer_bigrams, vectorizer_trigrams

def preprocess_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens

def extract_ngrams(tokens, n):
    return list(ngrams(tokens, n))

def custom_preprocessor(text):
    return ' '.join(['_'.join(bigram) for bigram in text])

def vectorize_text(tokens, vectorizer_tfidf, vectorizer_bigrams, vectorizer_trigrams):
    if not tokens:
        raise ValueError("Preprocessed text is empty after tokenization and stopword removal.")

    X_tfidf = vectorizer_tfidf.transform([' '.join(tokens)])
    
    bigrams = extract_ngrams(tokens, 2)
    trigrams = extract_ngrams(tokens, 3)
    
    X_bigrams = vectorizer_bigrams.transform([custom_preprocessor(bigrams)])
    X_trigrams = vectorizer_trigrams.transform([custom_preprocessor(trigrams)])
    
    X_combined = hstack((X_tfidf, X_bigrams, X_trigrams))
    return X_combined

def predict_message(model, message, vectorizer_tfidf, vectorizer_bigrams, vectorizer_trigrams):
    tokens = preprocess_text(message)
    if not tokens:
        return 'Unable to classify: input text is too short or contains only stopwords.'
    X_combined = vectorize_text(tokens, vectorizer_tfidf, vectorizer_bigrams, vectorizer_trigrams)
    prediction = model.predict(X_combined)
    return 'Spam' if prediction[0] == 1 else 'Ham'

# Streamlit app layout
st.sidebar.title("Instructions")
st.sidebar.write("""
1. Choose the model source from the dropdown.
2. **Upload model file**: Upload the trained model and vectorizers files.
3. **Load from GitHub**: Automatically load the model and vectorizers from the provided GitHub links.
4. Enter a message in the text area to classify it as Spam or Ham.
""")

st.title("Spam/Ham Email Message Classifier")

model, vectorizer_tfidf, vectorizer_bigrams, vectorizer_trigrams = load_model()

if model and vectorizer_tfidf and vectorizer_bigrams and vectorizer_trigrams:
    col1, col2, col3 = st.columns([0.5, 3, 0.5])  # Adjust the column widths for more space

    with col2:
        input_text = st.text_area("Enter the message to classify:", height=200)  # Increase the height of the text area
        if st.button("Classify"):
            if input_text:
                result = predict_message(model, input_text, vectorizer_tfidf, vectorizer_bigrams, vectorizer_trigrams)
                st.markdown(f"<h3 style='text-align: center;'>The message is classified as: {result}</h3>", unsafe_allow_html=True)  # Increase the size of the output text
            else:
                st.write("Please enter a message to classify.")


