
# 📰 Fake News Detection App

This project is a machine learning-powered **Fake News Detection** web application built using **Python** and **Streamlit**. It utilizes multiple classification models to determine whether a news article is fake or real.

---

## 🗃️ Dataset Description

We use two CSV files: `Fake.csv` and `True.csv`.

Each file contains the following columns:

- `title`: The headline of the news article
- `text`: The full content/body of the article
- `subject`: The topic/label the article belongs to (e.g., politics, news, world)
- `date`: The publishing date of the article

- `Fake.csv`: Contains fake news articles
- `True.csv`: Contains real news articles

> During preprocessing, only the `text` column is used for model training. The other columns are dropped.

---

## 🔍 Features

- Preprocessing of real-world news datasets (`Fake.csv` and `True.csv`)
- Model training with:
  - Logistic Regression
  - Decision Tree
  - Gradient Boosting
  - Random Forest
- Model accuracy evaluation
- Interactive Streamlit web app for live prediction
- Confusion matrix heatmap visualization

---

## 🧠 Model Accuracies

| Model               | Accuracy               |
|--------------------|------------------------|
| Logistic Regression| 0.9879679144385026     |
| Decision Tree      | 0.9954545454545455     |
| Gradient Boosting  | 0.9951871657754011     |
| Random Forest      | 0.9868983957219252     |

---

## 📈 Model Performance

### Logistic Regression
```
              precision    recall  f1-score   support
           0       0.99      0.99      0.99      5861
           1       0.99      0.99      0.99      5359
    accuracy                           0.99     11220
   macro avg       0.99      0.99      0.99     11220
weighted avg       0.99      0.99      0.99     11220
```

### Decision Tree
```
              precision    recall  f1-score   support
           0       1.00      0.99      1.00      5861
           1       0.99      1.00      1.00      5359
    accuracy                           1.00     11220
   macro avg       1.00      1.00      1.00     11220
weighted avg       1.00      1.00      1.00     11220
```

### Gradient Boosting
```
              precision    recall  f1-score   support
           0       1.00      0.99      1.00      5861
           1       0.99      1.00      0.99      5359
    accuracy                           1.00     11220
   macro avg       1.00      1.00      1.00     11220
weighted avg       1.00      1.00      1.00     11220
```

### Random Forest
```
              precision    recall  f1-score   support
           0       0.99      0.99      0.99      5861
           1       0.99      0.99      0.99      5359
    accuracy                           0.99     11220
   macro avg       0.99      0.99      0.99     11220
weighted avg       0.99      0.99      0.99     11220
```

---

## 🚀 Streamlit App

The model is deployed using **Streamlit** for interactive fake news prediction.

👉 [Launch the app here](https://fake-news1.streamlit.app/)

---

## ✅ Conclusion

This project demonstrates the successful application of **Natural Language Processing (NLP)** and **machine learning** techniques for detecting fake news. By training multiple classifiers like Logistic Regression, Decision Tree, Gradient Boosting, and Random Forest on real-world news data, we built a robust and interactive web app that can classify news articles as fake or real with significant accuracy.

The application also serves as an excellent foundation for understanding:
- Text preprocessing
- Feature extraction (TF-IDF)
- Model comparison
- Web deployment using Streamlit

---

## 🚀 Future Improvements

Here are several ways this project can be enhanced:

- **Real-time News Feed Integration**: Use APIs (e.g., NewsAPI) to analyze live news content.
- **Deep Learning Models**: Implement LSTM, BERT, or Transformer-based models for improved accuracy on long and complex texts.
- **Explainable AI (XAI)**: Add SHAP or LIME explanations to show why the model predicted a certain output.
- **User Feedback Loop**: Allow users to vote on the prediction accuracy and use the feedback for model improvement.
