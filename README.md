## Streamlit app documentation

Spam/Ham Message Classifier Streamlit App
Overview
This Streamlit app is designed to classify text messages as either "Spam" or "Ham" using a trained machine learning model. The model, along with necessary vectorizers, can be loaded either from local files or directly from GitHub.

**Access to web application:**  [Click here to access](https://ml-approach-to-email-spam-detection.streamlit.app/)


**Dependencies**
The app requires the following Python libraries:
*	streamlit: For building the web application interface.
*	pickle: For loading serialized model and vectorizer files.
*	string: For text processing tasks.
*	nltk: For natural language processing (tokenization, stopwords removal, lemmatization).
*	sklearn: For text vectorization and machine learning.
*	scipy: For handling sparse matrices.
*	requests: For downloading files from URLs.
*	io: For handling byte streams.

**NLTK Resources**
The app uses the Natural Language Toolkit (NLTK) for text preprocessing:
*	punkt: Tokenizer for splitting text into words.
*	stopwords: Common English stopwords to filter out.
*	wordnet: WordNet Lemmatizer for reducing words to their base forms.

**GitHub URLs**
The app provides predefined GitHub URLs for loading the model and vectorizers:
*	GITHUB_MODEL_URL: URL for the trained model file.
*	GITHUB_VECTOR_TFIDF_URL: URL for the TF-IDF vectorizer file.
*	GITHUB_VECTOR_BIGRAMS_URL: URL for the bigrams vectorizer file.
*	GITHUB_VECTOR_TRIGRAMS_URL: URL for the trigrams vectorizer file.

**Functions**  
``load_from_url(url)``  
Downloads and loads a pickle file from a given URL.  

``load_model()``  
Loads the model and vectorizers either from local files or from GitHub, based on user selection in the sidebar. Returns the model and vectorizers.  

``preprocess_text(text)``  
Cleans and preprocesses the input text:
*	Removes punctuation.
*	Tokenizes the text.
*	Converts tokens to lowercase.
*	Removes stopwords.
*	Applies lemmatization.  

``extract_ngrams(tokens, n)``  
Generates n-grams from a list of tokens.

``custom_preprocessor(text)``  
Custom preprocessing function for bigrams and trigrams, which joins n-grams with underscores.

``vectorize_text(tokens, vectorizer_tfidf, vectorizer_bigrams, vectorizer_trigrams)``  
Transforms preprocessed tokens into feature vectors using the provided vectorizers:
*	TF-IDF vectorizer.
*	Bigrams vectorizer.
*	Trigrams vectorizer.  

Combines the feature vectors into a single sparse matrix.

``predict_message(model, message, vectorizer_tfidf, vectorizer_bigrams, vectorizer_trigrams)``  
Classifies the input message as "Spam" or "Ham" using the provided model and vectorizers:
*	Preprocesses the input text.
*	Vectorizes the preprocessed text.
*	Uses the model to predict the category.

**Streamlit Layout**
*	Sidebar: Contains instructions and options to select the model source.
    o	Instructions for using the app.
    o	Options to upload files or load them from GitHub.
*	Main Area:
    o	Text area for entering the message to classify.
    o	Button to trigger classification and display the result.

**Usage**
1.	Instructions: Follow the steps provided in the sidebar to load the model and vectorizers.
2.	Upload Files: Choose to upload the model and vectorizers or load them from GitHub.
3.	Classify Message: Enter the message to classify in the text area and click the "Classify" button to see the result.

