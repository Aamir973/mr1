import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from sklearn.metrics import confusion_matrix, classification_report

# Load models and vectorizer
LR = joblib.load("lr_model.pkl")
DT = joblib.load("dt_model.pkl")
GB = joblib.load("gb_model.pkl")
RF = joblib.load("rf_model_compressed.pkl")  # Use compressed model
vectorizer = joblib.load("vectorizer.pkl")

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Prediction function
def predict_news(text):
    cleaned_text = preprocess_text(text)
    vector_input = vectorizer.transform([cleaned_text])
    predictions = {
        "Logistic Regression": LR.predict(vector_input)[0],
        "Decision Tree": DT.predict(vector_input)[0],
        "Gradient Boosting": GB.predict(vector_input)[0],
        "Random Forest (Compressed)": RF.predict(vector_input)[0],
    }
    return {model: 'Not Fake' if pred == 1 else 'Fake' for model, pred in predictions.items()}

# Streamlit UI
st.title("📰 Fake News Detection App")
st.write("Enter the news text below and find out if it's **real** or **fake**.")

user_input = st.text_area("📝 News Article Text", height=200)

if st.button("🔍 Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        results = predict_news(user_input)
        st.subheader("🔍 Model Predictions:")
        for model, result in results.items():
            st.write(f"**{model}**: {result}")

# Example Visualization (Static Confusion Matrix)
if st.checkbox("📊 Show Example Confusion Matrix"):
    y_true = [0, 1, 0, 1, 1, 0]
    y_pred = [0, 1, 1, 1, 1, 0]
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)
