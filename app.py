import streamlit as st
import numpy as np
import pandas as pd
import re
import time
import os
import requests
import json

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError as e:
    st.error(f"Gagal mengimpor transformers: {e}. Pastikan library transformers terinstall dengan versi terbaru.")
    st.error("Jalankan: pip install --upgrade transformers")
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

# Set page configuration as the first Streamlit command
st.set_page_config(layout="wide", page_icon="🛡️", page_title="Anti Hoax")

# Custom CSS for enhanced UI
st.markdown("""
<style>
body {
    font-family: 'Arial', sans-serif;
    background-color: #f0f2f6;
}
.stApp {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}
h1 {
    color: #1a73e8;
    text-align: center;
    font-size: 2.5em;
    margin-bottom: 10px;
}
h3 {
    color: #333;
    font-size: 1.5em;
    margin-top: 20px;
}
.stRadio > label, .stSelectbox > label, .stTextInput > label, .stTextArea > label {
    color: #444;
    font-weight: bold;
}
.stButton > button {
    background-color: #1a73e8;
    color: white;
    border-radius: 8px;
    padding: 10px 20px;
    font-size: 16px;
    transition: background-color 0.3s;
}
.stButton > button:hover {
    background-color: #1557b0;
}
.warning-box {
    background-color: #fff3cd;
    color: #856404;
    padding: 10px;
    border-radius: 5px;
    margin-top: 10px;
}
.error-box {
    background-color: #f8d7da;
    color: #721c24;
    padding: 10px;
    border-radius: 5px;
    margin-top: 10px;
}
.success-box {
    background-color: #d4edda;
    color: #155724;
    padding: 10px;
    border-radius: 5px;
    margin-top: 10px;
}
.reference-link {
    color: #1a73e8;
    text-decoration: none;
    font-size: 1.1em;
}
.reference-link:hover {
    text-decoration: underline;
}
.section-separator {
    border-bottom: 1px solid #e0e0e0;
    margin: 20px 0;
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
    st.error("API Key Jatevo tidak ditemukan di st.secrets. Tambahkan JATEVO_API_KEY di secrets.toml atau pengaturan Streamlit Cloud.")
    st.stop()

# Improved UI
st.title("🛡️ Anti Hoax Indonesia")
st.markdown("**Aplikasi deteksi hoaks berbasis AI untuk berita dalam Bahasa Indonesia.**")
st.markdown("Masukkan URL artikel atau teks berita untuk memeriksa apakah itu hoaks atau valid.")

# Cache model
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        model = AutoModelForSequenceClassification.from_pretrained("Rifky/indobert-hoax-classification", num_labels=2)
        base_model = SentenceTransformer("indobenchmark/indobert-base-p1")
        tokenizer = AutoTokenizer.from_pretrained("Rifky/indobert-hoax-classification", fast=True)
        data = load_dataset("Rifky/indonesian-hoax-news", split="train")
        return model, base_model, tokenizer, data
    except Exception as e:
        st.error(f"Gagal memuat model atau dataset: {e}")
        st.stop()

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Jatevo API query (no caching to ensure fresh responses)
def query_jatevo_hoax_explanation(text, prediction, confidence):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    prompt = f"""
    Analisis teks berikut untuk memverifikasi kebenaran faktualnya dalam konteks Indonesia. 
    Teks dianalisis sebagai {prediction} dengan tingkat kepercayaan {int(confidence*100)}%. 
    Berikan penjelasan singkat dan padat dalam Bahasa Indonesia maksimal 100 kata mengapa teks ini mungkin {prediction} atau salah secara faktual. 
    Fokus pada verifikasi fakta umum (misalnya, informasi resmi tentang Indonesia seperti geografi, sejarah, atau pemerintahan) 
    dan konteks budaya/sosial di Indonesia. Jika memungkinkan, gunakan informasi eksternal (misalnya, tren media sosial atau sumber terpercaya). 
    Jika teks mengandung klaim yang meragukan, soroti potensi kesalahan faktual.
    Teks: "{text[:500]}"  # 500 characters for context
    """
    
    payload = {
        "model": "deepseek-ai/DeepSeek-R1-0528",
        "messages": [{"role": "user", "content": prompt}],
        "stop": [],
        "stream": False,
        "top_p": 1,
        #"max_tokens": 500,  # Increased to 500 for more complete output
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
            # Remove <think> if it appears at the start
            if explanation.startswith("<think>"):
                explanation = explanation.replace("<think>", "").strip()
            return explanation
        return "Tidak ada penjelasan dari Jatevo API."
    except requests.exceptions.RequestException as e:
        return f"Error Jatevo API: {e}"

# UI Layout
input_column, reference_column = st.columns([3, 2])

with st.spinner("Memuat Model..."):
    model, base_model, tokenizer, data = load_model()

# Input options
with input_column:
    st.subheader("Masukkan Artikel")
    input_type = st.radio("Pilih jenis input:", ("URL Artikel", "Teks Langsung"))
    
    if input_type == "URL Artikel":
        user_input = st.text_input("URL Artikel", placeholder="https://example.com/berita", help="Masukkan URL artikel berita dalam Bahasa Indonesia.")
    else:
        user_input = st.text_area("Teks Artikel", placeholder="Masukkan teks artikel atau ringkasan...", height=150)
    
    process_option = st.selectbox("Proses teks:", ["Seluruh Artikel", "Hanya Judul", "Paragraf Pertama"])
    submit = st.button("Cek Hoaks")

# Process input
try:
    if submit and user_input:
        last_time = time.time()
        text = user_input
        title = ""

        if input_type == "URL Artikel":
            with st.spinner("Membaca Artikel..."):
                try:
                    scrape_result = scrape(user_input)
                    title, text = scrape_result.title, scrape_result.text
                except Exception as e:
                    st.error(f"Tidak dapat mengambil data artikel dari URL: {e}")
                    st.stop()
        
        if text:
            text = re.sub(r"\n", " ", text)

            if process_option == "Hanya Judul" and title:
                text = title
            elif process_option == "Paragraf Pertama":
                text = text.split(". ")[0] + "."

            with st.spinner("Menganalisis Hoaks..."):
                token = text.split()
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

                input_column.markdown(
                    f"<small>Analisis selesai dalam {int(time.time() - last_time)} detik</small>",
                    unsafe_allow_html=True,
                )
                if prediction:  # fake
                    input_column.markdown(f'<div class="error-box">Berita ini {prediction_label}.</div>', unsafe_allow_html=True)
                    input_column.markdown(f'<b>Tingkat Kepercayaan:</b> {int(confidence*100)}%', unsafe_allow_html=True)
                else:  # valid
                    input_column.markdown(f'<div class="success-box">Berita ini {prediction_label}.</div>', unsafe_allow_html=True)
                    input_column.markdown(f'<b>Tingkat Kepercayaan:</b> {int(confidence*100)}%', unsafe_allow_html=True)
                    if confidence < 0.7:  # Warn if confidence is low
                        input_column.markdown(
                            '<div class="warning-box">Keyakinan rendah. Disarankan untuk memeriksa fakta lebih lanjut dari sumber terpercaya seperti CekFakta.com atau media resmi.</div>',
                            unsafe_allow_html=True
                        )

                with st.spinner("Menghasilkan Penjelasan Generatif..."):
                    explanation = query_jatevo_hoax_explanation(text, prediction_label, confidence)
                    if explanation:
                        input_column.subheader("Penjelasan Generatif")
                        input_column.markdown(explanation)

                if input_type == "URL Artikel" and title:
                    with reference_column:
                        st.subheader("Artikel Referensi Terkait")
                        try:
                            title_embeddings = base_model.encode(title)
                            similarity_score = cosine_similarity([title_embeddings], data["embeddings"]).flatten()
                            sorted_indices = np.argsort(similarity_score)[::-1].tolist()
                            for i in sorted_indices[:5]:
                                st.markdown(
                                    f"""
                                    <small>{data["url"][i].split("/")[2]}</small>
                                    <a href="{data["url"][i]}" class="reference-link">{data["title"][i]}</a>
                                    """,
                                    unsafe_allow_html=True,
                                )
                        except Exception as e:
                            st.error(f"Gagal memuat artikel referensi: {e}")
    elif submit:
        st.error("Harap masukkan URL atau teks artikel.")
except Exception as e:
    st.error(f"Terjadi kesalahan: {e}")