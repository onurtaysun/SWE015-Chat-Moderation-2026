
import torch
import gradio as gr
import re
from transformers import AutoTokenizer

MODEL_PATH = "quantized_model.pt"
TOKENIZER_PATH = "moderation_tokenizer"

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\\S+|www\\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"(.)\\1{2,}", r"\\1\\1", text)
    text = re.sub(r"[^a-z0-9\\s!?']", " ", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
model = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
model.eval()

def predict_moderation(text):
    if not text or not text.strip():
        return {"Please enter a message": 1.0}

    cleaned_text = clean_text(text)
    inputs = tokenizer(
        cleaned_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=1)[0]

    return {
        "Clean": float(probabilities[0]),
        "Offensive": float(probabilities[1])
    }

demo = gr.Interface(
    fn=predict_moderation,
    inputs=gr.Textbox(lines=4, label="Enter Your Message"),
    outputs=gr.Label(num_top_classes=2, label="Moderation Results"),
    title="Real-Time Chat Moderation",
    description="Detect offensive messages using a Quantized Transformer model."
)

demo.launch()
