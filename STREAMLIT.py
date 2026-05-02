import streamlit as st
import torch
import torch.nn as nn
from transformers import BertTokenizer
import pickle
import pandas as pd
import re

# ==============================
# 1️⃣ ARCHITECTURE CLASSES
# ==============================
# 🔥 UPDATED: Matches the new anti-overfitting architecture
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        attn_weights = self.attn(lstm_output)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights * lstm_output, dim=1)
        return context, attn_weights

class CNN_BiLSTM_Attention(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embed_dropout = nn.Dropout(0.3)
        
        # 🔥 Shrunk to 64
        self.conv = nn.Conv1d(embed_dim, 64, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        
        # 🔥 Shrunk to 64
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            bidirectional=True,
            batch_first=True,
            dropout=0.4 # Higher dropout
        )
        
        # 🔥 Shrunk to 128 (64 * 2)
        self.attention = Attention(128)
        self.dropout = nn.Dropout(0.6)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        x = self.embed_dropout(x)

        # Apply Attention Mask
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1)

        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        
        lstm_out, _ = self.lstm(x)
        attn_out, weights = self.attention(lstm_out)
        
        out = self.dropout(attn_out)
        logits = self.fc(out)
        
        return logits, weights.squeeze(2)

# ==============================
# 2️⃣ CACHED LOADERS
# ==============================
@st.cache_resource
def load_assets():
    try:
        with open("label_encoder.pkl", "rb") as f:
            le = pickle.load(f)
    except FileNotFoundError:
        st.error("label_encoder.pkl not found!")
        st.stop()

    num_classes = len(le.classes_)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 🔥 UPDATED: embed_dim changed to 128
    model = CNN_BiLSTM_Attention(
        vocab_size=tokenizer.vocab_size,
        embed_dim=128, 
        num_classes=num_classes
    )
    
    try:
        model.load_state_dict(torch.load("best_model.pth", map_location=device))
    except FileNotFoundError:
        st.error("best_model.pth not found!")
        st.stop()
        
    model.to(device)
    model.eval()
    
    return model, tokenizer, le, device

model, tokenizer, le, device = load_assets()

# ==============================
# 3️⃣ TEXT CLEANING HELPER
# ==============================
def clean_input_text(text):
    """
    Cleans the text so it matches what the model saw during training.
    """
    text = text.lower() # Convert to lowercase
    text = re.sub(r'[^a-z0-9\s]', '', text) # Remove punctuation like ! ? , .
    return text.strip()

# ==============================
# 4️⃣ STREAMLIT UI
# ==============================
st.set_page_config(page_title="Mental Health Classifier", page_icon="🧠")

st.title("🧠 Mental Health Text Classifier")
st.markdown("Test your trained **CNN + BiLSTM + Attention** model in real-time.")

user_input = st.text_area("Enter text to classify:", height=150, placeholder="Type how you are feeling here...")

if st.button("Classify Text", type="primary"):
    if not user_input.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Analyzing..."):
            
            # 1. Clean the text FIRST
            cleaned_text = clean_input_text(user_input)
            
            # 2. Tokenize input
            inputs = tokenizer(
                cleaned_text,
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            
            # 3. Inference
            with torch.no_grad():
                logits, attention_weights = model(input_ids, attention_mask=attention_mask)
                probabilities = torch.softmax(logits, dim=1).squeeze()
                predicted_class_idx = torch.argmax(probabilities).item()
                
            # 4. Decode prediction mapping
            LABEL_MAP = {
                0: "Anxiety",
                1: "Bipolar",
                2: "Depression",
                3: "Normal",
                4: "Personality disorder",
                5: "Stress",
                6: "Suicidal"
            }
            
            predicted_label = LABEL_MAP.get(predicted_class_idx, "Unknown")
            confidence = probabilities[predicted_class_idx].item() * 100
            
            # --- RESULTS ---
            st.success(f"**Prediction:** {predicted_label}")
            st.info(f"**Confidence:** {confidence:.2f}%")
            
            # --- ATTENTION VISUALIZATION ---
            with st.expander("🔍 View Attention Weights"):
                st.markdown("See which words the model focused on the most.")
                
                tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
                weights = attention_weights[0].cpu().numpy()
                
                pool_factor = len(tokens) // len(weights)
                mapped_tokens = [tokens[i * pool_factor] for i in range(len(weights))]
                
                attn_df = pd.DataFrame({
                    "Token": mapped_tokens,
                    "Weight": weights
                })
                
                attn_df = attn_df[attn_df["Token"] != "[PAD]"]
                st.bar_chart(attn_df.set_index("Token"))