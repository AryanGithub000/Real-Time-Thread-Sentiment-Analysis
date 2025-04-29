import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from collections import Counter
import shap
import os

st.set_page_config(layout="wide")

# Load Models and Tokenizer
MODEL_1_PATH = r"C:\Users\admin\Desktop\Twitter Sentiment Analysis\To be shared\new_lstm_model.h5"
MODEL_2_PATH = "model.h5"
TOKENIZER_PATH = "tokenizer.pkl"

if not os.path.exists(MODEL_1_PATH) or not os.path.exists(MODEL_2_PATH):
    st.error("One or both model files are missing!")
    st.stop()

model_1 = tf.keras.models.load_model(MODEL_1_PATH)
model_2 = tf.keras.models.load_model(MODEL_2_PATH)

with open(TOKENIZER_PATH, "rb") as handle:
    tokenizer = pickle.load(handle)

MAX_SEQUENCE_LENGTH = 100
CSV_FILE = "twentyfive_thirty.csv"
tweet_data = pd.read_csv(CSV_FILE, encoding="ISO-8859-1")

def preprocess_tweets(texts):
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding="post")

def analyze_sentiment_with_lstm(thread_df, model):
    processed_tweets = preprocess_tweets(thread_df["text"].tolist())
    predictions = model.predict(processed_tweets, batch_size=32)
    sentiment_labels = np.argmax(predictions, axis=1)
    confidences = np.max(predictions, axis=1)
    sentiment_map = {0: "Sadness", 1: "Joy", 2: "Love", 3: "Anger", 4: "Fear", 5: "Surprise"}
    thread_df["Sentiment"] = [sentiment_map[label] for label in sentiment_labels]
    thread_df["Confidence"] = confidences
    thread_sentiment = Counter(thread_df["Sentiment"]).most_common(1)[0][0]
    return thread_df[["text", "Sentiment", "Confidence"]], thread_sentiment

def plot_sentiment_distribution(sentiment_df, ax, title):
    sns.countplot(x="Sentiment", data=sentiment_df, palette="coolwarm", ax=ax)
    ax.set_title(title)

def plot_sentiment_trend(sentiment_df, ax, title):
    sentiment_df["Index"] = range(len(sentiment_df))
    sns.lineplot(x="Index", y="Confidence", hue="Sentiment", data=sentiment_df, marker="o", ax=ax)
    ax.set_title(title)

# Create Sidebar Tabs
tabs = st.sidebar.radio("Select an Option:", [
    "Single Tweet Sentiment Analyzer",
    "Tweet Thread Sentiment Analyzer",
    "Other Feature!"
])

if tabs == "Single Tweet Sentiment Analyzer":
    st.title("📌 Single Tweet Sentiment Analyzer")
    # Load tokenizer
    with open("tokenizer.pkl", "rb") as handle:
        tokenizer = pickle.load(handle)

    # Load models
    gru_model = tf.keras.models.load_model("new_lstm_keras_model.keras")
    lstm_model = tf.keras.models.load_model("model.h5")

    st.title("Twitter Sentiment Analysis")
    st.markdown("Multiclass classification (6 classes) using **Bi-directional GRU and LSTM**.")

    def preprocess_text(text, tokenizer, max_length=100):
        """Tokenize and pad text for GRU/LSTM models."""
        sequences = tokenizer.texts_to_sequences([text])
        return pad_sequences(sequences, maxlen=max_length, padding="post")

    input_text = st.text_area("Enter a tweet:", placeholder="Write Something ....")

    if st.button("Analyze"):
        if input_text:
            gru_input = preprocess_text(input_text, tokenizer)
            lstm_input = preprocess_text(input_text, tokenizer)
            
            gru_probs = gru_model.predict(gru_input)[0]
            lstm_probs = lstm_model.predict(lstm_input)[0]
            
            class_labels = ["Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"]
            
            st.subheader("Predictions")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**GRU Model Predictions:**")
                probs_df_gru = pd.DataFrame({"Class": class_labels, "Probability": gru_probs * 100})
                st.dataframe(probs_df_gru)
                st.bar_chart(probs_df_gru.set_index("Class"))
            
            with col2:
                st.write("**LSTM Model Predictions:**")
                probs_df_lstm = pd.DataFrame({"Class": class_labels, "Probability": lstm_probs * 100})
                st.dataframe(probs_df_lstm)
                st.bar_chart(probs_df_lstm.set_index("Class"))

            st.subheader("Word Importance Analysis")
            st.markdown(f"**Input text:** :blue-background[{input_text}]")
            
            explainer_gru = shap.KernelExplainer(gru_model.predict, np.zeros((100, 100)))
            shap_values_gru = explainer_gru.shap_values(gru_input)
            
            explainer_lstm = shap.KernelExplainer(lstm_model.predict, np.zeros((100, 100)))
            shap_values_lstm = explainer_lstm.shap_values(lstm_input)
            
            shap_values_filtered_gru = [value for i, value in enumerate(shap_values_gru[0][0]) if gru_input[0, i] != 0]
            shap_values_filtered_lstm = [value for i, value in enumerate(shap_values_lstm[0][0]) if lstm_input[0, i] != 0]
            
            feature_names = [tokenizer.index_word[i] for i in gru_input[0] if i != 0]
            
            colx, coly = st.columns(2)
            with colx:
                st.write("**GRU Model Word Importance:**")
                values = np.array(shap_values_filtered_gru).reshape(1, -1)
                mean_shap_values = np.abs(values).mean(axis=0)
                sorted_indices = np.argsort(mean_shap_values)
                sorted_features = [feature_names[i] for i in sorted_indices]
                sorted_values = mean_shap_values[sorted_indices]
                
                plt.figure(figsize=(12, 6))
                plt.barh(sorted_features, sorted_values, color='skyblue', edgecolor='black')
                plt.xlabel("Mean SHAP Value")
                plt.ylabel("Feature Words")
                plt.title("GRU Model Feature Importance")
                plt.gca().invert_yaxis()
                plt.grid(axis='x', linestyle='--', alpha=0.6)
                st.pyplot(plt.gcf())
            
            with coly:
                st.write("**LSTM Model Word Importance:**")
                values = np.array(shap_values_filtered_lstm).reshape(1, -1)
                mean_shap_values = np.abs(values).mean(axis=0)
                sorted_indices = np.argsort(mean_shap_values)
                sorted_features = [feature_names[i] for i in sorted_indices]
                sorted_values = mean_shap_values[sorted_indices]
                
                plt.figure(figsize=(12, 6))
                plt.barh(sorted_features, sorted_values, color='lightcoral', edgecolor='black')
                plt.xlabel("Mean SHAP Value")
                plt.ylabel("Feature Words")
                plt.title("LSTM Model Feature Importance")
                plt.gca().invert_yaxis()
                plt.grid(axis='x', linestyle='--', alpha=0.6)
                st.pyplot(plt.gcf())
        else:
            st.markdown("*:red[Please write something!]*")

elif tabs == "Tweet Thread Sentiment Analyzer":
    st.title("🚀 Emotion Detection Comparison in Twitter Threads")
    thread_number = st.text_input("🔎 Enter a Thread Number:")

    if st.button("Analyze Sentiment"):
        if thread_number:
            with st.spinner("Fetching and analyzing conversation..."):
                df = tweet_data[tweet_data["thread_number"] == thread_number]
                if df.empty:
                    st.error("No conversation found for this thread number.")
                else:
                    col1, col2 = st.columns(2)
                    with col1:
                        sentiment_df_1, overall_sentiment_1 = analyze_sentiment_with_lstm(df.copy(), model_1)
                        st.subheader(f"Model 1: Thread Sentiment - **{overall_sentiment_1}**")
                        st.dataframe(sentiment_df_1)
                        fig, ax = plt.subplots(figsize=(6, 4))
                        plot_sentiment_distribution(sentiment_df_1, ax, "Model 1: Sentiment Distribution")
                        st.pyplot(fig)
                        fig, ax = plt.subplots(figsize=(8, 5))
                        plot_sentiment_trend(sentiment_df_1, ax, "Model 1: Sentiment Evolution")
                        st.pyplot(fig)
                    with col2:
                        sentiment_df_2, overall_sentiment_2 = analyze_sentiment_with_lstm(df.copy(), model_2)
                        st.subheader(f"Model 2: Thread Sentiment - **{overall_sentiment_2}**")
                        st.dataframe(sentiment_df_2)
                        fig, ax = plt.subplots(figsize=(6, 4))
                        plot_sentiment_distribution(sentiment_df_2, ax, "Model 2: Sentiment Distribution")
                        st.pyplot(fig)
                        fig, ax = plt.subplots(figsize=(8, 5))
                        plot_sentiment_trend(sentiment_df_2, ax, "Model 2: Sentiment Evolution")
                        st.pyplot(fig)
        else:
            st.warning("⚠️ Please enter a valid Thread Number.")

elif tabs == "Other Feature!":
    st.title("🛠 Other Feature!")
    st.write("Coming soon!")
