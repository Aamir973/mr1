import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import shap
import lime
import numpy as np
from sklearn.metrics import confusion_matrix
import streamlit.components.v1 as components
from lime.lime_text import LimeTextExplainer

# Load models and vectorizer
LR = joblib.load("lr_model.pkl")
DT = joblib.load('dt_model.pkl')
GB = joblib.load('gb_model.pkl')
RF = joblib.load("rf_model_compressed.pkl")  # Corrected name
vectorizer = joblib.load('vectorizer.pkl')

# Preprocessing
def wordopt(text):
    text = text.lower()
    text = re.sub('$.*?$', '', text)
    text = re.sub("\W", " ", text)
    text = re.sub('https?://\S+|[www.\S+](http://www.\S+)', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('\[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

def predict_all_models(news):
    clean_text = wordopt(news)
    vector_input = vectorizer.transform([clean_text])
    results = {
        "Logistic Regression": "Not Fake" if LR.predict(vector_input)[0] == 1 else "Fake",
        "Decision Tree": "Not Fake" if DT.predict(vector_input)[0] == 1 else "Fake",
        "Gradient Boosting": "Not Fake" if GB.predict(vector_input)[0] == 1 else "Fake",
        "Random Forest": "Not Fake" if RF.predict(vector_input)[0] == 1 else "Fake"
    }
    return results

# Streamlit UI
st.title("📰 Fake News Detection App with SHAP & LIME")
st.write("Enter the news text below and find out if it's **real** or **fake**.")

user_input = st.text_area("News Article Text")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        predictions = predict_all_models(user_input)
        for model, result in predictions.items():
            st.write(f"**{model}**: {result}")

        # SHAP Explanation
        if st.checkbox("Show SHAP Explanation (Logistic Regression)"):
            st.subheader("SHAP Force Plot (HTML Rendered)")
            clean_text = wordopt(user_input)
            vector_input = vectorizer.transform([clean_text])

            explainer = shap.LinearExplainer(LR, vectorizer.transform(["sample text"]), feature_perturbation="interventional")
            shap_values = explainer.shap_values(vector_input)

            # SHAP force plot rendered as HTML
            shap.initjs()
            force_plot = shap.force_plot(
                explainer.expected_value,
                shap_values[0],
                feature_names=vectorizer.get_feature_names_out(),
                matplotlib=False
            )
            html_shap = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
            components.html(html_shap, height=400)

        # LIME Explanation
        if st.checkbox("Show LIME Explanation (Logistic Regression)"):
            st.subheader("LIME Explanation")
            lime_explainer = LimeTextExplainer(class_names=["Fake", "Not Fake"])
            lime_exp = lime_explainer.explain_instance(
                user_input,
                lambda x: LR.predict_proba(vectorizer.transform([wordopt(i) for i in x])),
                num_features=10
            )
            components.html(lime_exp.as_html(), height=600, scrolling=True)

# Confusion matrix
if st.checkbox("Show Example Confusion Matrix"):
    y_true = [0, 1, 0, 1, 1, 0]
    y_pred = [0, 1, 1, 1, 1, 0]
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)
