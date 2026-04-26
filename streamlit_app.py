import streamlit as st
import tensorflow as tf
import pickle
import re
import string
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model
model = tf.keras.models.load_model("sentiment_model.h5")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_LEN = 200

# Text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r"\d+", "", text)
    return text

# UI
st.set_page_config(page_title="Sentiment Analysis", layout="centered")

st.title("🎬 Sentiment Analysis App")
st.write("Enter a movie review to predict sentiment")

user_input = st.text_area("✍️ Enter your review:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        cleaned = clean_text(user_input)
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')

        prediction = model.predict(padded)[0][0]

        if prediction > 0.5:
            st.success(f"Positive 😊 (Confidence: {prediction:.2f})")
        else:
            st.error(f"Negative 😡 (Confidence: {prediction:.2f})")