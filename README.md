# Real-Time-Thread-Sentiment-Analysis

#1. GRU Model Training Notebook:
https://colab.research.google.com/drive/1oksXes9K2fFKewggy0IQBy-FKhG5jQaC#scrollTo=VBKDbsBcOiWh

#2. LSTM Model Training Notebook:
https://colab.research.google.com/drive/1CX7uwHE3pvTIMS9jiXIFyxgCCxZCU2eT#scrollTo=CsGUu_8kf7_s

#3. BERT Model Training Notebook:
https://colab.research.google.com/drive/1LpChlvXFZgubRd2PYMibZ1bIz1FrnGgW#scrollTo=pWEdxNOQPGTB

# Twitter Sentiment Analysis Multi-Model Comparison

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)](https://streamlit.io/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/transformers/)

A comprehensive Twitter sentiment analysis application using multiple deep learning models (Bidirectional GRU, Bidirectional LSTM, and TinyBERT) for emotion classification across 6 categories: Sadness, Joy, Love, Anger, Fear, and Surprise.

![WhatsApp Image 2025-04-26 at 16 17 04_2f4c9c8b](https://github.com/user-attachments/assets/427b029c-50c4-4cc8-83f4-3e152624552f)


## Features

### 1. Single Tweet Analysis
- Input any tweet text and get sentiment predictions from three different models
- Compare prediction probabilities across models
- Visualize word importance using SHAP (SHapley Additive exPlanations)
- View detailed model comparison metrics

### 2. Tweet Thread Analysis
- Analyze entire Twitter conversation threads
- Visualize sentiment distribution and trends
- Compare model predictions across multiple tweets
- Identify the most influential words in sentiment detection

### 3. Live Twitter API Integration
- Fetch and analyze real-time tweets using Twitter API
- Track sentiment evolution in threads
- Compare model performance on live data

## Models Used

### 1. Bidirectional GRU
- Custom-trained deep learning model for 6-class emotion detection
- Optimized for tweet-specific language patterns

### 2. Bidirectional LSTM
- Sequential model specialized in capturing long-term dependencies
- Trained on social media text data

### 3. TinyBERT
- Compressed BERT model fine-tuned for emotion classification
- Leverages Huawei Noah's TinyBERT_General_4L_312D architecture
- 4 transformer layers with 312 hidden dimensions
- 12 attention heads

## Technical Implementation

### Model Comparison Metrics
- **Cosine Similarity:** Measures how similar probability distributions are between models
- **KL Divergence:** Quantifies how one probability distribution differs from another
- **Model Agreement Analysis:** Identifies when models reach consensus or disagree

### Visualization Tools
- Probability distribution bar charts
- Radar charts for multi-model comparison
- Sentiment evolution timelines
- Word importance analysis

## Required Files

Ensure you have the following files in your project directory:

- `new_lstm_keras_model.keras`: BiGRU model
- `model.h5`: BiLSTM model
- `tokenizer.pkl`: Pre-trained tokenizer
- `twentyfive_thirty.csv`: Sample Twitter data
- `BERT/` directory containing:
  - `config.json`
  - `special_tokens_map.json`
  - `tokenizer_config.json`
  - Model weights

## Twitter API Setup

To use the Twitter API features:
1. Create a developer account at [Twitter Developer Portal](https://developer.twitter.com/en/portal/dashboard)
2. Generate an API bearer token
3. Replace the `BEARER_TOKEN` variable in the code with your token

## Usage Guide

### Single Tweet Analysis
1. Select "Single Tweet Sentiment Analyzer" from the sidebar
2. Enter a tweet in the text area
3. Click "Analyze Sentiment"
4. View the sentiment predictions and word importance visualization

### Thread Analysis
1. Select "Tweet Thread Sentiment Analyzer" from the sidebar
2. Enter a thread number
3. Click "Analyze Sentiment"
4. Compare the sentiment distributions and evolution across models

### Live Twitter Analysis
1. Select "Tweet Sentiment Analyzer (Twitter API)" from the sidebar
2. Enter a Tweet ID
3. Click "Analyze Sentiment"
4. Explore the thread's sentiment analysis across all models

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- TinyBERT implementation based on [Huawei Noah's Ark Lab](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT)
- SHAP library for model interpretability
- Streamlit for the interactive web application

## Current Needs:
There seems to be a problem to run on Hugging Face deployment; can test it locally
