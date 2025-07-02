import streamlit as st
import nltk
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# NLTK setup
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

# === Load and preprocess text ===
def preprocess_text(text):
    sentences = sent_tokenize(text)
    processed = []
    stop_words = set(stopwords.words('english'))
    for sentence in sentences:
        clean = re.sub(r'\W+', ' ', sentence.lower())
        filtered = ' '.join([word for word in clean.split() if word not in stop_words])
        processed.append(filtered)
    return sentences, processed

# === Find most relevant sentence ===
def get_most_relevant_sentence(user_input, original_sentences, processed_sentences):
    if not user_input.strip():
        return "❗ Please enter a valid question."

    vectorizer = TfidfVectorizer()
    all_sentences = processed_sentences + [user_input]
    tfidf_matrix = vectorizer.fit_transform(all_sentences)

    # Compare user_input (last vector) to all others
    similarities = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])
    index = similarities.argmax()
    score = similarities[0][index]

    if score < 0.2:
        return "🤔 I couldn't find anything relevant. Try rephrasing."
    return original_sentences[index]

# === Chatbot logic ===
def chatbot(user_input, original_sentences, processed_sentences):
    return get_most_relevant_sentence(user_input, original_sentences, processed_sentences)

# === Streamlit App ===
def main():
    st.set_page_config(page_title="Custom Text Chatbot", layout="centered")
    st.title("📚 Custom Knowledge Chatbot")
    st.write("Upload a text file based on your topic. Then ask your question!")

    uploaded_file = st.file_uploader("📄 Upload your .txt file", type=["txt"])

    if uploaded_file:
        raw_text = uploaded_file.read().decode('utf-8')
        original_sentences, processed_sentences = preprocess_text(raw_text)

        user_input = st.text_input("💬 Ask your question:")
        if user_input:
            response = chatbot(user_input, original_sentences, processed_sentences)
            st.subheader("🤖 Chatbot says:")
            st.write(response)

    else:
        st.info("Upload a .txt file to activate the chatbot.")

if __name__ == "__main__":
    main()
