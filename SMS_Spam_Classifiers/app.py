import streamlit as st
import pandas as pd
import numpy as np
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

# Load the model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('spam_model.pkl', 'rb'))

# Download NLTK stopwords once
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

# Text preprocessing function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    temp = []
    for word in text:
        if word.isalnum():
            temp.append(word)

    text = temp[:]
    temp.clear()

    for word in text:
        if word not in stopwords.words('english') and word not in string.punctuation:
            temp.append(ps.stem(word))

    return " ".join(temp)  # <-- return a string, not a list

# Streamlit UI
st.title("SMS Spam Classifier")

input_sms = st.text_area("Enter your message:", height=200)

if st.button('Predict'):
    # 1. Preprocess
    transformed_sms = transform_text(input_sms)

    # 2. Vectorize
    vector_input = tfidf.transform([transformed_sms])

    # 3. Predict
    result = model.predict(vector_input)[0]

    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
