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
import snscrape.modules.twitter as sntwitter
import tweepy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.spatial.distance import cosine
from scipy.stats import entropy
from collections import Counter
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # Added for BERT
import torch 

st.set_page_config(layout="wide")

# Load Models and Tokenizer
MODEL_1_PATH = "new_lstm_keras_model.keras"
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
tweet_data = pd.read_csv(CSV_FILE, encoding="ISO-8859-1")\

sentiment_map = {0: "Sadness", 1: "Joy", 2: "Love", 3: "Anger", 4: "Fear", 5: "Surprise"}

def preprocess_tweets(texts):
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding="post")

def analyze_sentiment_with_lstm(thread_df, model):
    processed_tweets = preprocess_tweets(thread_df["text"].tolist())
    predictions = model.predict(processed_tweets, batch_size=32)
    sentiment_labels = np.argmax(predictions, axis=1)
    confidences = np.max(predictions, axis=1)
    
    thread_df["Sentiment"] = [sentiment_map[label] for label in sentiment_labels]
    thread_df["Confidence"] = confidences
    thread_sentiment = Counter(thread_df["Sentiment"]).most_common(1)[0][0]
    return thread_df[["text", "Sentiment", "Confidence"]], thread_sentiment
    import torch
import numpy as np
from collections import Counter

# ---------------------------------- BERT --------------------------------------
BERT_MODEL_PATH = "./BERT"  # Path to BERT model directory

tokenizer_bert = AutoTokenizer.from_pretrained(BERT_MODEL_PATH)
model_bert = AutoModelForSequenceClassification.from_pretrained(
                                                                BERT_MODEL_PATH, 
                                                                num_labels=6,  # Changed for 6 classes
                                                                trust_remote_code=True,
                                                                ignore_mismatched_sizes=True
                                                            )
sentiment_map_bert = {
    0: "Sadness", 1: "Joy", 2: "Love",
    3: "Anger", 4: "Fear", 5: "Surprise"
}


def bert_predict(text):
    inputs = tokenizer_bert(text, 
                          return_tensors="pt", 
                          padding=True, 
                          truncation=True, 
                          max_length=512)
    
    with torch.no_grad():
        outputs = model_bert(**inputs)
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probs.numpy()[0], sentiment_map_bert[torch.argmax(outputs.logits).item()]

import torch
import numpy as np
from collections import Counter

import torch
import numpy as np
from collections import Counter

def analyze_sentiment_with_bert(thread_df, tokenizer_bert, model_bert):
    # Define sentiment mapping for 6-class classification
    sentiment_map_bert = {0: "Sadness", 1: "Joy", 2: "Love", 3: "Anger", 4: "Fear", 5: "Surprise"}
    
    # Store results
    sentiments = []
    confidences = []

    # Ensure model is in evaluation mode
    model_bert.eval()

    # Iterate through each tweet separately
    for text in thread_df["text"].tolist():
        # Tokenize the individual tweet
        inputs = tokenizer_bert(text, 
                                return_tensors="pt", 
                                padding=True, 
                                truncation=True, 
                                max_length=512)
        
        # Get model predictions
        with torch.no_grad():
            outputs = model_bert(**inputs)

        # Compute softmax probabilities
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()[0]

        # Determine sentiment label and confidence score
        sentiment_label = np.argmax(probs)
        confidence = np.max(probs)

        # Append results
        sentiments.append(sentiment_map_bert[sentiment_label])
        confidences.append(confidence)

    # Add results to the DataFrame
    thread_df["Sentiment"] = sentiments
    thread_df["Confidence"] = confidences

    # Determine the most common sentiment in the thread
    thread_sentiment = Counter(thread_df["Sentiment"]).most_common(1)[0][0]

    return thread_df[["text", "Sentiment", "Confidence"]], thread_sentiment

# ------------------------------------------------------- PLOTS ----------------------------------------------------
def plot_sentiment_distribution(sentiment_df, ax, title):
    sns.countplot(x="Sentiment", data=sentiment_df, palette="coolwarm", ax=ax)
    ax.set_title(title)

def plot_sentiment_trend(sentiment_df, ax, title):
    sentiment_df["Index"] = range(len(sentiment_df))
    sns.lineplot(x="Index", y="Confidence", hue="Sentiment", data=sentiment_df, marker="o", ax=ax)
    ax.set_title(title)

# --------------------------------------- COMPARISON METRICS ---------------------------------------------------------
# Function to compute metrics
def calculate_similarity_metrics(probs1, probs2):
    cosine_sim = 1 - cosine(probs1, probs2)
    kl_div = entropy(probs1, probs2)
    return cosine_sim, kl_div

def analyze_tweet(input_text):
    processed_input = preprocess_text(input_text, tokenizer)
    probs_model_1 = model_1.predict(processed_input)[0]
    probs_model_2 = model_2.predict(processed_input)[0]
    probs_model_3, pred_label_3 = bert_predict(input_text)
    
    pred_label_1 = class_labels[np.argmax(probs_model_1)]
    pred_label_2 = class_labels[np.argmax(probs_model_2)]
    
    cosine_sim, kl_div = calculate_similarity_metrics(probs_model_1, probs_model_2)
    agreement = pred_label_1 == pred_label_2
    
    return probs_model_1, probs_model_2, probs_model_3, pred_label_1, pred_label_2,  pred_label_3, cosine_sim, kl_div, agreement

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

def plot_model_comparison(comparison_df):

    col1, col2 = st.columns(2)  # Create two columns for side-by-side plots

    # Grouped Bar Chart
    with col1:
        st.markdown("📊 Model Probability Comparison (Grouped Bar Chart)")
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(comparison_df["Sentiment"]))
        width = 0.35

        ax.bar(x - width/2, comparison_df["Model 1 Probability"], width, label='Model 1', color='skyblue')
        ax.bar(x + width/2, comparison_df["Model 2 Probability"], width, label='Model 2', color='salmon')
        ax.bar(x + width/2, comparison_df["Model 3 Probability"], width, label='Model 3', color='red')


        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df["Sentiment"], rotation=45)
        ax.set_ylabel("Probability")
        ax.set_title("Model Probability Comparison")
        ax.legend()
        st.pyplot(fig)

        # Explanation of Insights from the Bar Chart
        st.markdown("🔍 **Insight:**")
        st.markdown(""":blue-background[Higher bars indicate stronger model confidence.
                       Similar bars suggest model agreement; large differences show disagreement.
                       Helps identify which model is more confident in specific sentiments.]""")

    # Radar Chart
    with col2:
        st.markdown("📈 Model Probability Spread (Radar Chart)")
        labels = comparison_df["Sentiment"].tolist()
        n_labels = len(labels)
        angles = np.linspace(0, 2 * np.pi, n_labels, endpoint=False).tolist()

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.set_xticks(angles)
        ax.set_xticklabels(labels)

        values_1 = comparison_df["Model 1 Probability"].tolist()
        values_2 = comparison_df["Model 2 Probability"].tolist()
        values_3 = comparison_df["Model 3 Probability"].tolist()

        values_1 += values_1[:1]
        values_2 += values_2[:1]
        values_3 += values_3[:1]

        angles += angles[:1]

        ax.plot(angles, values_1, label='Model 1', color='blue')
        ax.fill(angles, values_1, color='blue', alpha=0.3)
        ax.plot(angles, values_2, label='Model 2', color='red')
        ax.fill(angles, values_2, color='red', alpha=0.3)
        ax.plot(angles, values_3, label='Model 3', color='green')
        ax.fill(angles, values_3, color='green', alpha=0.3)
        ax.legend()
        st.pyplot(fig)

        # Explanation of Insights from the Radar Chart
        st.markdown("🔍 **Insights:**")
        st.markdown(""":blue-background[Higher bars indicate stronger model confidence. 
                    Similar bars suggest model agreement; large differences show disagreement.
                     Helps identify which model is more confident in specific sentiments.]""")

def display_metrics(probs_1, probs_2, probs_3, label_1, label_2, label_3, cosine_sim, kl_div, agreement):
    comparison_df = pd.DataFrame({
        "Sentiment": class_labels,
        "Model 1 Probability": probs_1 * 100,
        "Model 2 Probability": probs_2 * 100,
        "Model 3 Probability": probs_3 * 100

    })
    st.markdown("##")
    st.subheader("1. Model Comparison plots -")
    with st.container(border = True):
        plot_model_comparison(comparison_df)
    
    st.markdown("##")
    st.subheader("2. Model Comparison Metrics -")
    colx, coly, colz = st.columns([1,1,1])

    
    with colx:
        with st.container(border = True):
            st.write("**:green[Model Agreement]:**", "✅ Yes" if agreement else "❌ No")
            st.markdown("##")
            st.markdown("*:blue-background[Model agreement shows how often both models predict the same sentiment. A high agreement means both models are making similar predictions.]*")

                                
    with coly:
        with st.container(border = True):
            st.write(f"**:green[Cosine Similarity]:** {cosine_sim:.4f}")
            st.markdown("##")
            st.markdown("*:blue-background[Cosine similarity measures how similar the probability distributions of both models are. A score close to 1 means the models are highly aligned in their predictions.]*")

    with colz:
        with st.container(border = True):
            st.write(f"**:green[KL Divergence]:** {kl_div:.4f}")
            st.markdown("##")
            st.markdown("*:blue-background[KL divergence measures how much one probability distribution differs from another. A lower value means the models are making similar predictions, while a higher value indicates disagreement.]*")
        


# --------------------------------------------------- 3rd Feature using Twitter API ----------------------------------------------
# BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAN2EzgEAAAAAZb4min%2FzC5HpWHBNQMl17oCBhck%3Dg2U2L2uZbrmxRScq2KYRDeORIgidP4R80BIQjBSW0zo3NnoY1G"
# BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAN2EzgEAAAAAu4BPlY%2B8uu4SxMeSUxw2P9MlDts%3DzwlK5lGeCMGuxaLQwDOmDky3DAuHPDHUkctiTjuXhvLr7GKIHp"

BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAM8ryQEAAAAAC95kdjrtK4SsC4%2F6znorAdsRFK4%3DnxLIITqNg8DI7PfhyonbOI3eDKLX58UziPDZkW6DAAgBOH4040"

# Authenticate Twitter API
client = tweepy.Client(bearer_token=BEARER_TOKEN, wait_on_rate_limit=True)

def fetch_thread_from_tweet_id(tweet_id):
    """
    Fetches a full Twitter thread starting from the given tweet ID.
    Returns a DataFrame with tweet details.
    """
    try:
        # Fetch the root tweet
        tweet = client.get_tweet(tweet_id, tweet_fields=["conversation_id", "author_id", "created_at", "text"])
        if not tweet.data:
            st.error("Tweet not found or unavailable!")
            return pd.DataFrame(), None

        # Extract conversation ID (this identifies the full thread)
        conversation_id = tweet.data["conversation_id"]
        author_id = tweet.data["author_id"]

        # Fetch all tweets in the thread (same conversation ID)
        query = f"conversation_id:{conversation_id} from:{author_id}"
        response = client.search_recent_tweets(query=query, tweet_fields=["created_at", "text"])

        if not response.data:
            st.error("No thread found for this Tweet ID!")
            return pd.DataFrame(), None

        # Convert to DataFrame
        thread_data = [
            {"tweet_id": t.id, "created_at": t.created_at, "text": t.text}
            for t in response.data
        ]
        df = pd.DataFrame(thread_data).sort_values(by="created_at")

        return df, conversation_id

    except tweepy.TweepyException as e:
        st.error(f"Twitter API error: {e}")
        return pd.DataFrame(), None

# Load tokenizer
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

# Load models
gru_model = tf.keras.models.load_model("new_lstm_keras_model.keras")
lstm_model = tf.keras.models.load_model("model.h5")

st.title("Twitter Sentiment Analysis")
st.markdown("Multiclass classification (6 classes) using **:blue-background[Bi-directional GRU], :blue-background[Bi-directional LSTM] and :blue-background[miniBERT]**.")

def preprocess_text(text, tokenizer, max_length=100):
    """Tokenize and pad text for GRU/LSTM models."""
    sequences = tokenizer.texts_to_sequences([text])
    return pad_sequences(sequences, maxlen=max_length, padding="post")

# Create Sidebar Tabs
tabs = st.sidebar.radio("Select an Option:", [
    "Single Tweet Sentiment Analyzer",
    "Tweet Thread Sentiment Analyzer",
    "Tweet Sentiment Analyzer (Twitter API)"
])

# Define class_labels globally
class_labels = ["Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"]

if tabs == "Single Tweet Sentiment Analyzer":
    st.title("📌 Single Tweet Sentiment Analyzer")
    
    input_text = st.text_area("Enter a tweet:", placeholder="Write Something ....")

    if st.button("Analyze Sentiment"):
        if input_text:
            gru_input = preprocess_text(input_text, tokenizer)
            lstm_input = preprocess_text(input_text, tokenizer)
            

            gru_probs = gru_model.predict(gru_input)[0]
            lstm_probs = lstm_model.predict(lstm_input)[0]
            bert_probs, bert_label  = bert_predict(input_text)
            
            st.subheader("Predictions")
            col1, col2, col3 = st.columns(3)
            
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
            
            with col3:
                st.write("**BERT Model Predictions:**")
                probs_df_bert = pd.DataFrame({"Class": class_labels, "Probability": bert_probs * 100})
                st.dataframe(probs_df_bert)
                st.bar_chart(probs_df_bert.set_index("Class"))
            
            st.divider()
            st.subheader("Word Importance Analysis")
            st.markdown(f"**Input text:** :blue-background[{input_text}]")
            
            explainer_gru = shap.KernelExplainer(gru_model.predict, np.zeros((100, 100)))
            shap_values_gru = explainer_gru.shap_values(gru_input)
                        
            shap_values_filtered_gru = [value for i, value in enumerate(shap_values_gru[0][0]) if gru_input[0, i] != 0]

            feature_names = [tokenizer.index_word[i] for i in gru_input[0] if i != 0]


            st.write("**Combined Models - Word Importance:**")
            values = np.array(shap_values_filtered_gru).reshape(1, -1)
            mean_shap_values = np.abs(values).mean(axis=0)
            sorted_indices = np.argsort(mean_shap_values)
            sorted_features = [feature_names[i] for i in sorted_indices]
            sorted_values = mean_shap_values[sorted_indices]
            
            plt.figure(figsize=(12, 6))
            plt.barh(sorted_features, sorted_values, color='skyblue', edgecolor='black')
            plt.xlabel("Mean SHAP Value")
            plt.ylabel("Feature Words")
            plt.title("Combined Models - Feature Importance")
            plt.gca().invert_yaxis()
            plt.grid(axis='x', linestyle='--', alpha=0.6)
            st.pyplot(plt.gcf())
            
            
            # ----------------- Predictions for Metric based comparison ------------------------
            st.divider()
            probs_1, probs_2, probs_3, label_1, label_2, label_3, cosine_sim, kl_div, agreement = analyze_tweet(input_text)
            display_metrics(probs_1, probs_2, probs_3, label_1, label_2,  label_3, cosine_sim, kl_div, agreement)
        
        else:
            st.markdown("*:red[Please write something!]*")

elif tabs == "Tweet Thread Sentiment Analyzer":
    st.title("🚀 Sentiment Comparison for Twitter Threads")

    thread_number = st.text_input("🔎 Enter a Thread Number:")

    # Initialize session state variables
    if "analyzed" not in st.session_state:
        st.session_state.analyzed = False
    if "selected_text" not in st.session_state:
        st.session_state.selected_text = None

    if st.button("Analyze Sentiment") or st.session_state.analyzed:
        if thread_number:
            with st.spinner("Fetching and analyzing conversation..."):
                df = tweet_data[tweet_data["thread_number"] == thread_number]

                if df.empty:
                    st.error("No conversation found for this thread number.")
                    st.session_state.analyzed = False
                else:
                    # Store analyzed dataframe and set analyzed flag
                    st.session_state.analyzed_df = df
                    st.session_state.analyzed = True
                    st.session_state.selected_text = df["text"].iloc[0]  # Reset selection

                    col1, col2, col3 = st.columns(3)
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

                    with col3:
                        sentiment_df_3, overall_sentiment_3 = analyze_sentiment_with_bert(df.copy(), tokenizer_bert, model_bert)
                        st.subheader(f"Model 3: Thread Sentiment - **{overall_sentiment_3}**")
                        st.dataframe(sentiment_df_3)
                        fig, ax = plt.subplots(figsize=(6, 4))
                        plot_sentiment_distribution(sentiment_df_3, ax, "Model 3: Sentiment Distribution")
                        st.pyplot(fig)
                        fig, ax = plt.subplots(figsize=(8, 5))
                        plot_sentiment_trend(sentiment_df_3, ax, "Model 3: Sentiment Evolution")
                        st.pyplot(fig)
        else:
            st.warning("⚠️ Please enter a valid Thread Number.")
            st.session_state.analyzed = False
    
    # ========== WORD IMPORTANCE ANALYSIS ==========
    # Word Importance Section (outside the button block)
    if st.session_state.analyzed:
        st.divider()
        st.subheader("Word Importance Analysis")

        # Make sure we have the dataframe and it has the text column
        if hasattr(st.session_state, 'analyzed_df') and 'text' in st.session_state.analyzed_df.columns:
            df = st.session_state.analyzed_df
            
            # Get current selection index
            current_texts = df["text"].tolist()
            current_index = 0
            if st.session_state.selected_text in current_texts:
                current_index = current_texts.index(st.session_state.selected_text)
            
            # Update selection widget
            selected_text = st.selectbox(
                "Select a text",
                current_texts,
                index=current_index,
                key="text_selector"
            )
            
            # Update session state
            st.session_state.selected_text = selected_text

            if selected_text:
                gru_input = preprocess_text(selected_text, tokenizer)
                explainer_gru = shap.KernelExplainer(gru_model.predict, np.zeros((100, 100)))
                shap_values_gru = explainer_gru.shap_values(gru_input)

                shap_values_filtered_gru = [value for i, value in enumerate(shap_values_gru[0][0]) if gru_input[0, i] != 0]
                feature_names = [tokenizer.index_word[i] for i in gru_input[0] if i != 0]

                st.write("**Combined Models - Word Importance:**")
                values = np.array(shap_values_filtered_gru).reshape(1, -1)
                mean_shap_values = np.abs(values).mean(axis=0)
                sorted_indices = np.argsort(mean_shap_values)
                sorted_features = [feature_names[i] for i in sorted_indices]
                sorted_values = mean_shap_values[sorted_indices]

                plt.figure(figsize=(12, 6))
                plt.barh(sorted_features, sorted_values, color='skyblue', edgecolor='black')
                plt.xlabel("Mean SHAP Value")
                plt.ylabel("Feature Words")
                plt.title("Combined Models - Feature Importance")
                plt.gca().invert_yaxis()
                plt.grid(axis='x', linestyle='--', alpha=0.6)
                st.pyplot(plt.gcf())
            else:
                st.write(":red[Select a text!]")
        else:
            st.warning("No text data available for analysis.")

elif tabs == "Tweet Sentiment Analyzer (Twitter API)":
    st.title("🚀 Sentiment Comparison for Twitter Threads (Twitter API)")

    tweet_id = st.text_input("🔎 Enter a Tweet ID:")

    if st.button("Analyze Sentiment"):
        if tweet_id:
            with st.spinner("Fetching and analyzing conversation..."):
                df, thread_id = fetch_thread_from_tweet_id(tweet_id)
                
                if df.empty:
                    st.error("No conversation found for this Tweet ID.")
                else:
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        sentiment_df_1, overall_sentiment_1 = analyze_sentiment_with_lstm(df.copy(), model_1)
                        st.subheader(f"GRU: Thread Sentiment - **{overall_sentiment_1}**")
                        st.dataframe(sentiment_df_1)
                        
                        fig, ax = plt.subplots(figsize=(6, 4))
                        plot_sentiment_distribution(sentiment_df_1, ax, "GRU: Sentiment Distribution")
                        st.pyplot(fig)

                        fig, ax = plt.subplots(figsize=(8, 5))
                        plot_sentiment_trend(sentiment_df_1, ax, "GRU: Sentiment Evolution")
                        st.pyplot(fig)

                    with col2:
                        sentiment_df_2, overall_sentiment_2 = analyze_sentiment_with_lstm(df.copy(), model_2)
                        st.subheader(f"LSTM: Thread Sentiment - **{overall_sentiment_2}**")
                        st.dataframe(sentiment_df_2)
                        
                        fig, ax = plt.subplots(figsize=(6, 4))
                        plot_sentiment_distribution(sentiment_df_2, ax, "LSTM: Sentiment Distribution")
                        st.pyplot(fig)

                        fig, ax = plt.subplots(figsize=(8, 5))
                        plot_sentiment_trend(sentiment_df_2, ax, "LSTM: Sentiment Evolution")
                        st.pyplot(fig)
                    with col3:
                        sentiment_df_3, overall_sentiment_3 = analyze_sentiment_with_bert(df.copy(), tokenizer_bert, model_bert)
                        st.subheader(f"miniBERT: Thread Sentiment - **{overall_sentiment_3}**")
                        st.dataframe(sentiment_df_3)
                        fig, ax = plt.subplots(figsize=(6, 4))
                        plot_sentiment_distribution(sentiment_df_3, ax, "miniBERT: Sentiment Distribution")
                        st.pyplot(fig)
                        fig, ax = plt.subplots(figsize=(8, 5))
                        plot_sentiment_trend(sentiment_df_3, ax, "miniBERT: Sentiment Evolution")
                        st.pyplot(fig)

                    # Only show word importance analysis if we have valid data
                    st.divider()
                    st.subheader("Word Importance Analysis")
                    
                    # Check if df has data and the 'text' column exists
                    if not df.empty and 'text' in df.columns:
                        text = st.selectbox("select a text", df["text"], placeholder="select a text")
                        if text:
                            gru_input = preprocess_text(text, tokenizer)
                            explainer_gru = shap.KernelExplainer(gru_model.predict, np.zeros((100, 100)))
                            shap_values_gru = explainer_gru.shap_values(gru_input)
                                        
                            shap_values_filtered_gru = [value for i, value in enumerate(shap_values_gru[0][0]) if gru_input[0, i] != 0]

                            feature_names = [tokenizer.index_word[i] for i in gru_input[0] if i != 0]
                            st.write("**Combined Models - Word Importance:**")
                            values = np.array(shap_values_filtered_gru).reshape(1, -1)
                            mean_shap_values = np.abs(values).mean(axis=0)
                            sorted_indices = np.argsort(mean_shap_values)
                            sorted_features = [feature_names[i] for i in sorted_indices]
                            sorted_values = mean_shap_values[sorted_indices]
                            
                            plt.figure(figsize=(12, 6))
                            plt.barh(sorted_features, sorted_values, color='skyblue', edgecolor='black')
                            plt.xlabel("Mean SHAP Value")
                            plt.ylabel("Feature Words")
                            plt.title("Combined Models - Feature Importance")
                            plt.gca().invert_yaxis()
                            plt.grid(axis='x', linestyle='--', alpha=0.6)
                            st.pyplot(plt.gcf())
                        else:
                            st.write(":red[Select a text!]")
                    else:
                        st.warning("No text data available for analysis.")
        else:
            st.warning("⚠️ Please enter a valid Tweet ID.")