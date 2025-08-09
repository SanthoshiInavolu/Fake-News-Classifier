import streamlit as st
import joblib

model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
def predict_news(text):
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]
    return "Fake News" if prediction==1 else "Real News"

st.set_page_config(page_title = "Fake News Detector" ,page_icon="📰",layout="centered")
st.title("📰 Fake News Detector")
st.write("Type or paste a news headline / article below to check if it's Fake or Real. ")
user_input = st.text_area("News Text " ,placeholder = "Enter news content here... ")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text before predicting. ")
    else:
        result = predict_news(user_input)
        if result == "Fake News":
            st.error(f"Prediction : {result}")
        else:
            st.success(f"Prediction : {result}")