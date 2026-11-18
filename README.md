# Real-Time Chat Moderation

## Project Description
An intelligent application for detecting offensive comments using machine learning and transformer models.

## Models
1. TF-IDF + Logistic Regression
2. Word2Vec + Feedforward Neural Network
3. DistilBERT
4. Quantized DistilBERT

## Model Performance

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| TF-IDF + Logistic Regression | 0.9386 | 0.8472 |
| Word2Vec + Neural Network | 0.9534 | 0.8487 |
| DistilBERT | 0.9597 | 0.8905 |
| Quantized DistilBERT | 0.9640 | 0.8941 |

## Quantization
Model size was reduced from 255.45 MB to 132.29 MB, achieving a 48.21% reduction.

## Application
The application uses Gradio and Quantized DistilBERT for real-time chat moderation.

## Technologies
Python, PyTorch, Scikit-learn, Gensim, Hugging Face Transformers, Gradio.

## Installation
Run: pip install -r requirements.txt

## Dataset
Jigsaw Toxic Comment Classification Dataset.

## Project
SWE015
