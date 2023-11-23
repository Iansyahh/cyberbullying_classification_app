# -*- coding: utf-8 -*-
import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer

# Load your machine learning model
with open('cyberbully_detection.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the vectorizer used during training
with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Define the Streamlit app
def main():
    st.title("Cyberbullying Detection App")

    # Input for the user to enter text
    user_input = st.text_area("Enter text for cyberbullying detection:", "Type here...")

    if st.button("Predict"):
        # Perform the prediction
        X_new = vectorizer.transform([user_input])  # Use the same vectorizer used during training
        prediction = model.predict(X_new)

        # Display the result
        if prediction[0] == 0:
            st.success("Prediction: Not Cyberbullying")
        else:
            st.error("Prediction: Cyberbullying")

if __name__ == "__main__":
    main()
