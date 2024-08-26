import streamlit as st
import torch
from tensorflow.keras.preprocessing import sequence
import pickle
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
import re

model = torch.jit.load('sentiment_model.pt', map_location=torch.device('cpu'))

with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

stop_words = set(stopwords.words('english'))
voc_size = 10000
max_len = 500

def preprocess_text(text):
    text = re.sub(r'<[^>]+>', '', text)
    text = text.lower()
    cleaned_text = re.sub(r'[^\w\s]', '', text)
    tokens = cleaned_text.split()
    filtered_words = [word for word in tokens if word not in stop_words]
    processed_text = ' '.join(filtered_words)
    return processed_text

st.sidebar.header("Sentiment Analysis Instructions")
st.sidebar.write(
    """
    1. Enter a movie or product review in the text box.
    2. Click the **Predict** button.
    3. The app will predict whether the sentiment is positive or negative.
    """
)

st.title('Sentiment Analysis Web App 📝')
st.write("Analyze the sentiment of your review. The model classifies the sentiment as either Positive or Negative.")

user_input = st.text_area("Enter your review here:")

if st.button('Predict'):
    if user_input:
        processed_input = preprocess_text(user_input)
        processed_input = tokenizer.texts_to_sequences([processed_input])[0]
        input_padded = sequence.pad_sequences([processed_input], maxlen=max_len)
        input_tensor = torch.tensor(input_padded, dtype=torch.long)
        
        with torch.no_grad():
            output = model(input_tensor)
            prediction = (output > 0.5).float()
            sentiment = 'Positive' if prediction.item() == 1 else 'Negative'
            if sentiment == 'Positive':
                st.success(f'The sentiment of the review is: **{sentiment}**')
            else:
                st.error(f'The sentiment of the review is: **{sentiment}**')
    else:
        st.warning("Please enter a review.")