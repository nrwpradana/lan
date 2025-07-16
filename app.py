import streamlit as st
import numpy as np
import pandas as pd
import re
import time
import requests
import json
try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError as e:
    st.error(f"Gagal mengimpor transformers: {e}. Pastikan library transformers terinstall dengan versi terbaru.")
    st.error("Jalankan: pip install --upgrade transformers torch")
    st.stop()
try:
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError as e:
    st.error(f"Gagal mengimpor library: {e}. Install dengan: pip install datasets sentence-transformers scikit-learn")
    st.stop()
try:
    from src.scraper import scrape
except ImportError as e:
    st.error(f"Gagal mengimpor scraper: {e}. Pastikan modul src.scraper tersedia.")
    st.stop()

# Set page configuration
st.set_page_config(layout="wide", page_icon="🛡️", page_title="Anti Hoax Chat")

# Custom CSS for Grok-like UI
st.markdown("""
<style>
body {
    font-family: 'Segoe UI', sans-serif;
    background-color: #1e1e2f;
    color: #ffffff;
}
.stApp {
    max-width: 1000px;
    margin: 0 auto;
    padding: 20px;
}
h1 {
    color: #00d4ff;
    text-align: center;
    font-size: 2.8em;
    font-weight: 700;
    margin-bottom: 10px;
}
.chat-container {
    background-color: #2a2a3d;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    max-height: 500px;
    overflow-y: auto;
}
.chat-message {
    padding: 12px;
    margin: 8px 0;
    border-radius: 8px;
    max-width: 80%;
}
.user-message {
    background-color: #007bff;
    color: white;
    margin-left: auto;
}
.bot-message {
    background-color: #3a3a4f;
    color: #ffffff;
}
.stTextInput > div > div > input {
    border-radius: 8px;
    padding: 12px;
    border: 1px solid #00d4ff;
    background-color: #2a2a3d;
    color: #ffffff;
}
.stButton > button {
    background-color: #00d4ff;
    color: #1e1e2f;
    border-radius: 8px;
    padding: 10px 20px;
    font-size: 16px;
    transition: background-color 0.3s;
}
.stButton > button:hover {
    background-color: #00aaff;
}
.result-box {
    padding: 15px;
    border-radius: 8px;
    margin-top: 15px;
}
.valid-box {
    background-color: #2a4b2a;
    color: #ffffff;
}
.fake-box {
    background-color: #4b2a2a;
    color: #ffffff;
}
.warning-box {
    background-color: #4b4b2a;
    color: #ffffff;
    padding: 15px;
    border-radius: 8px;
    margin-top: 15px;
}
.reference-link {
    color: #00d4ff;
    text-decoration: none;
}
.reference-link:hover {
    text-decoration: underline;
}
.section-separator {
    border-bottom: 1px solid #3a3a4f;
    margin: 20px 0;
}
.spinner {
    border: 4px solid #3a3a4f;
    border-top: 4px solid #00d4ff;
    border-radius: 50%;
    width: 24px;
    height: 24px;
    animation: spin 1s linear infinite;
    margin: 10px auto;
}
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
</style>
""", unsafe_allow_html=True)

# Jatevo API configuration
BASE_URL = "https://inference.jatevo.id/v1"
ENDPOINT = f"{BASE_URL}/chat/completions"

# Label dictionary
label = {0: "valid", 1: "fake"}

# Load API key from st.secrets
try:
    API_KEY = st.secrets["JATEVO_API_KEY"]
except KeyError:
    st.error("API Key Jatevo tidak ditemukan di st.secrets. Tambahkan JATEVO_API_KEY di secrets.toml.")
    st.stop()

# Cache model
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        model = AutoModelForSequenceClassification.from_pretrained("Rifky/indobert-hoax-classification", num_labels=2)
        base_model = SentenceTransformer("firqaaa/indo-sentence-bert-base")
        tokenizer = AutoTokenizer.from_pretrained("Rifky/indobert-hoax-classification", use_fast=True)
        data = load_dataset("Rifky/indonesian-hoax-news", split="train")
        if "embeddings" not in data.column_names:
            st.warning("Kolom 'embeddings' tidak ditemukan. Mengencode ulang judul...")
            titles = data["title"]
            embeddings = base_model.encode(titles, convert_to_tensor=True)
            data = data.add_column("embeddings", embeddings.tolist())
        return model, base_model, tokenizer, data
    except Exception as e:
        st.error(f"Gagal memuat model atau dataset: {e}")
        st.stop()

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Jatevo API query
def query_jatevo_hoax_explanation(text, prediction, confidence):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    prompt = f"""
    Analisis judul berikut untuk memverifikasi kebenaran faktualnya dalam konteks Indonesia. 
    Judul dianalisis sebagai {prediction} dengan tingkat kepercayaan {int(confidence*100)}%. 
    Berikan penjelasan singkat, padat, dan jelas dalam 100 kata Bahasa Indonesia mengapa judul ini mungkin {prediction} atau salah secara faktual. 
    Gunakan informasi eksternal jika memungkinkan (misalnya, tren media sosial atau sumber terpercaya). 
    Soroti potensi kesalahan faktual jika ada.
    Judul: "{text}"
    """
    payload = {
        "model": "deepseek-ai/DeepSeek-R1-0528",
        "messages": [{"role": "user", "content": prompt}],
        "stop": [],
        "stream": False,
        "top_p": 0.95,
        "top_k": 50,
        "temperature": 0.7,
        "presence_penalty": 0,
        "frequency_penalty": 0
    }
    try:
        response = requests.post(ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()
        json_data = response.json()
        if 'choices' in json_data and len(json_data['choices']) > 0:
            explanation = json_data['choices'][0]['message']['content']
            if explanation.startswith("<think>"):
                explanation = explanation.replace("<think>", "").strip()
            return explanation
        return "Tidak ada penjelasan dari Jatevo API."
    except requests.exceptions.RequestException as e:
        return f"Error Jatevo API: {e}"

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'submitted' not in st.session_state:
    st.session_state.submitted = False

# UI Layout
st.title("🛡️ Anti Hoax Indonesia")
st.markdown("**Chat dengan AI untuk deteksi hoaks berita dalam Bahasa Indonesia.**")
st.markdown("Masukkan judul berita atau URL artikel, dan saya akan memeriksa apakah itu hoaks atau valid!")

# Reset chat history button
if st.button("Hapus Riwayat"):
    st.session_state.chat_history = []
    st.session_state.submitted = False
    st.rerun()

# Chat container
chat_container = st.container()
with chat_container:
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            st.markdown(f'<div class="chat-message user-message">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message bot-message">{message["content"]}</div>', unsafe_allow_html=True)

# Input form
with st.form(key='chat_form', clear_on_submit=True):
    user_input = st.text_input("Masukkan judul berita atau URL:", placeholder="Contoh: 'Vaksin menyebabkan kemandulan' atau URL artikel")
    submit = st.form_submit_button("Cek Hoaks")

# Process input
if submit and user_input and not st.session_state.submitted:
    st.session_state.submitted = True
    st.markdown('<div class="spinner"></div>', unsafe_allow_html=True)
    
    # Load model
    with st.spinner("Memuat Model..."):
        model, base_model, tokenizer, data = load_model()
    
    # Add user message
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    with st.spinner("Menganalisis Hoaks..."):
        last_time = time.time()
        text = user_input
        title = user_input

        # Check if input is a URL
        if user_input.startswith(('http://', 'https://')):
            try:
                scrape_result = scrape(user_input)
                title = scrape_result.title if hasattr(scrape_result, 'title') else user_input
                if not title:
                    title = user_input
            except Exception as e:
                st.session_state.chat_history.append({
                    "role": "bot",
                    "content": f"Error: Tidak dapat mengambil judul dari URL: {e}"
                })
                st.session_state.submitted = False
                st.rerun()
                st.stop()

        # Process title
        token = title.split()
        text_len = len(token)
        sequences = []
        for i in range(text_len // 512):
            sequences.append(" ".join(token[i * 512 : (i + 1) * 512]))
        sequences.append(" ".join(token[text_len - (text_len % 512) : text_len]))
        sequences = tokenizer(
            sequences,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        predictions = model(**sequences)[0].detach().numpy()
        result = [
            np.sum([sigmoid(i[0]) for i in predictions]) / len(predictions),
            np.sum([sigmoid(i[1]) for i in predictions]) / len(predictions),
        ]

        prediction = np.argmax(result, axis=-1)
        confidence = result[prediction]
        prediction_label = label[prediction]

        # Bot response
        bot_response = f"""
        Analisis selesai dalam {int(time.time() - last_time)} detik.<br>
        <div class="result-box {'valid-box' if prediction == 0 else 'fake-box'}">
        Berita ini {prediction_label}.<br>
        <b>Tingkat Kepercayaan:</b> {int(confidence*100)}%
        </div>
        """
        if confidence < 0.7:
            bot_response += """
            <div class="warning-box">
            Keyakinan rendah. Disarankan memeriksa fakta lebih lanjut di CekFakta.com atau media resmi.
            </div>
            """

        # Generate explanation
        with st.spinner("Menghasilkan Penjelasan..."):
            explanation = query_jatevo_hoax_explanation(title, prediction_label, confidence)
            if explanation:
                bot_response += f"<h4>Penjelasan</h4>{explanation}"

        # Reference articles
        bot_response += "<h4>Artikel Referensi Terkait</h4>"
        try:
            if "embeddings" not in data.column_names:
                bot_response += "Kolom 'embeddings' tidak ditemukan. Tidak dapat menampilkan referensi."
            else:
                title_embeddings = base_model.encode([title])[0]
                similarity_score = cosine_similarity([title_embeddings], data["embeddings"]).flatten()
                sorted_indices = np.argsort(similarity_score)[::-1].tolist()
                if sorted_indices:
                    for i in sorted_indices[:5]:
                        bot_response += f"""
                        <small>{data['url'][i].split('/')[2] if 'url' in data.column_names else 'Sumber Tidak Tersedia'}</small><br>
                        <a href="{data['url'][i] if 'url' in data.column_names else '#'}" class="reference-link">{data['title'][i]}</a><br>
                        """
                else:
                    bot_response += "Tidak ada referensi yang relevan ditemukan."
        except Exception as e:
            bot_response += f"Gagal memuat artikel referensi: {e}<br>Tidak ada referensi tersedia."

        # Add bot response
        st.session_state.chat_history.append({"role": "bot", "content": bot_response})

    st.session_state.submitted = False
    st.rerun()

# Handle empty input
if submit and not user_input:
    st.session_state.chat_history.append({
        "role": "bot",
        "content": "Harap masukkan judul berita atau URL artikel."
    })
    st.session_state.submitted = False
    st.rerun()