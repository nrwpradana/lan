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
    st.error(f"Gagal mengimpor scraper: {e}. Pastikan modul src.preprocessor.scraper tersedia.")
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
.inconsistency-box {
    background-color: #fff3cd;
    color: #856404;
    padding: 10px;
    border-radius: 5px;
    margin-top: 10px;
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

# Cache model
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        model = AutoModelForSequenceClassification.from_pretrained("Rifky/indobert-hoax-classification", num_labels=2)
        base_model = SentenceTransformer("indobenchmark/indobert-base-p1")
        tokenizer = AutoTokenizer.from_pretrained("Rifky/indobert-hoax-classification", fast=True)
        data = load_dataset("Rifky/indonesian-hoax-news", split="train")
        # Pre-compute embeddings for RAG
        if "embeddings" not in data.column_names:
            titles = data["title"]
            embeddings = base_model.encode(titles, convert_to_tensor=True).cpu().numpy()
            data = data.add_column("embeddings", embeddings.tolist())
        return model, base_model, tokenizer, data
    except Exception as e:
        st.error(f"Gagal memuat model atau dataset: {e}")
        st.stop()

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# RAG-enhanced Jatevo query
def query_jatevo_rag(text, prediction, confidence, base_model, data):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    # Retrieve relevant context using similarity search
    text_embedding = base_model.encode(text, convert_to_tensor=True).cpu().numpy()
    similarity_score = cosine_similarity([text_embedding], data["embeddings"]).flatten()
    sorted_indices = np.argsort(similarity_score)[::-1].tolist()
    relevant_context = ""
    for i in sorted_indices[:3]:  # Top 3 most similar items
        relevant_context += f"Judul: {data['title'][i]}, Teks: {data['text'][i][:200]}...\n"

    prompt = f"""
    Analisis teks berikut untuk memverifikasi kebenaran faktualnya dalam konteks Indonesia. 
    Teks dianalisis sebagai {prediction} dengan tingkat kepercayaan {int(confidence*100)}%. 
    Gunakan konteks berikut untuk mendukung analisis: {relevant_context}
    Berikan penjelasan singkat, padat, dan jelas DALAM MAKSIMAL 100 KATA Bahasa Indonesia mengapa teks ini mungkin {prediction} atau salah secara faktual. 
    Hindari kalimat berbelit-belit. Soroti kesalahan faktual jika ada klaim meragukan. 
    Teks: "{text[:600]}"
    """

    payload = {
        "model": "deepseek-ai/DeepSeek-R1-0528",
        "messages": [{"role": "user", "content": prompt}],
        "stop": [],
        "stream": False,
        "max_tokens": 150,  # Batasi panjang respons (sekitar 100-120 kata)
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
            # Batasi ke 100 kata
            words = explanation.split()
            limited_explanation = " ".join(words[:100]) if len(words) > 100 else explanation
            return limited_explanation
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
                            '<div class="warning-box">Keyakinan rendah. Disarankan untuk memeriksa fakta lebih lanjut.</div>',
                            unsafe_allow_html=True
                        )

                with st.spinner("Menghasilkan Penjelasan Generatif dengan RAG..."):
                    explanation = query_jatevo_rag(text, prediction_label, confidence, base_model, data)
                    if explanation:
                        input_column.subheader("Penjelasan Generatif")
                        input_column.markdown(explanation)
                        # Check for inconsistency
                        if "fake" in explanation.lower() and prediction_label == "valid":
                            input_column.markdown(
                                '<div class="inconsistency-box">Peringatan: Model valid, tapi RAG menunjukkan kemungkinan hoaks. Verifikasi lebih lanjut dianjurkan.</div>',
                                unsafe_allow_html=True
                            )
                        elif "valid" in explanation.lower() and prediction_label == "fake":
                            input_column.markdown(
                                '<div class="inconsistency-box">Peringatan: Model hoaks, tapi RAG menunjukkan valid. Verifikasi lebih lanjut dianjurkan.</div>',
                                unsafe_allow_html=True
                            )

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