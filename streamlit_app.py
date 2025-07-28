import streamlit as st
import pickle

# Load model and vectorizer
with open("logistic_regression1_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Streamlit UI
st.title("🚨 Disaster Tweet Classifier")
st.markdown("Enter a tweet below to predict whether it is about a **real disaster** or **not**.")

# User input
tweet = st.text_area("Tweet Input", "")

# Prediction
if st.button("Classify"):
    if tweet.strip() == "":
        st.warning("Please enter a tweet.")
    else:
        vec = vectorizer.transform([tweet])
        pred = model.predict(vec)[0]
        label = "🆘 Disaster" if pred == 1 else "✅ Not a Disaster"
        st.success(f"Prediction: **{label}**")
