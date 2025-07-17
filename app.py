import streamlit as st
import numpy as np
import pandas as pd
import re
import time
import os
import requests
import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.scraper import scrape

# Set page configuration
st.set_page_config(layout="wide", page_icon="🛡️", page_title="Anti Hoax")

# Custom CSS
st.markdown("""
<style>
body { font-family: 'Arial', sans-serif; background-color: #f0f2f6; }
.stApp { max-width: 1200px; margin: 0 auto; padding: 20px; }
h1 { color: #1a73e8; text-align: center; font-size: 2.5em; margin-bottom: 10px; }
h3 { color: #333; font-size: 1.5em; margin-top: 20px; }
.stTextInput > label { color: #444; font-weight: bold; }
.stButton > button { background-color: #1a73e8; color: white; border-radius: 8px; padding: 10px 20px; font-size: 16px; transition: background-color 0.3s; }
.stButton > button:hover { background-color: #1557b0; }
.warning-box { background-color: #fff3cd; color: #856404; padding: 10px; border-radius: 5px; margin-top: 10px; }
.error-box { background-color: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px; margin-top: 10px; }
.success-box { background-color: #d4edda; color: #155724; padding: 10px; border-radius: 5px; margin-top: 10px; }
.reference-link { color: #1a73e8; text-decoration: none; font-size: 1.1em; }
.reference-link:hover { text-decoration: underline; }
.section-separator { border-bottom: 1px solid #e0e0e0; margin: 20px 0; }
</style>
""", unsafe_allow_html=True)

# Jatevo API configuration
BASE_URL = "https://inference.jatevo.id/v1"
ENDPOINT = f"{BASE_URL}/chat/completions"
label = {0: "valid", 1: "fake"}

# Load API key
try:
    API_KEY = st.secrets["JATEVO_API_KEY"]
except KeyError:
    st.error("API Key Jatevo tidak ditemukan di st.secrets.")
    st.stop()

# Cache model
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        model = AutoModelForSequenceClassification.from_pretrained("Rifky/indobert-hoax-classification", num_labels=2)
        base_model = SentenceTransformer("firqaaa/indo-sentence-bert-base")
        tokenizer = AutoTokenizer.from_pretrained("Rifky/indobert-hoax-classification", fast=True)
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
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
    prompt = f"""
    Analisis judul berikut untuk memverifikasi kebenaran faktualnya dalam konteks Indonesia. 
    Judul dianalisis sebagai {prediction} dengan tingkat kepercayaan {int(confidence*100)}%. 
    Berikan penjelasan singkat dalam 100 kata Bahasa Indonesia mengapa judul ini mungkin {prediction}. 
    Gunakan informasi eksternal jika memungkinkan. Soroti potensi kesalahan faktual.
    Judul: "{text[:500]}"
    """
    payload = {
        "model": "deepseek-ai/DeepSeek-R1-0528",
        "messages": [{"role": "user", "content": prompt}],
        "stop": [], "stream": False, "top_p": 0.95, "top_k": 50, "temperature": 0.7,
        "presence_penalty": 0, "frequency_penalty": 0
    }
    try:
        response = requests.post(ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()
        json_data = response.json()
        explanation = json_data['choices'][0]['message']['content']
        if explanation.startswith("<think>"):
            explanation = explanation.replace("<think>", "").strip()
        return explanation
    except requests.exceptions.RequestException as e:
        return f"Error Jatevo API: {e}"

# UI Layout
input_column, reference_column = st.columns([3, 2])

with st.spinner("Memuat Model..."):
    model, base_model, tokenizer, data = load_model()

# Input URL
with input_column:
    st.subheader("Masukkan URL Artikel")
    user_input = st.text_input("URL Artikel", placeholder="https://example.com/berita")
    submit = st.button("Cek Hoaks")

# Process input
try:
    if submit and user_input:
        last_time = time.time()
        with st.spinner("Membaca Judul Artikel..."):
            try:
                scrape_result = scrape(user_input)
                title = scrape_result.title if hasattr(scrape_result, 'title') else ""
                if not title:
                    st.error("Judul artikel tidak ditemukan dari URL.")
                    st.stop()
            except Exception as e:
                st.error(f"Tidak dapat mengambil judul artikel dari URL: {e}")
                st.stop()

        with st.spinner("Menganalisis Hoaks..."):
            text = re.sub(r"\n", " ", title)
            sequences = tokenizer(
                text, max_length=512, truncation=True, padding="max_length", return_tensors="pt"
            )
            predictions = model(**sequences)[0].detach().numpy()
            result = [sigmoid(predictions[0][0]), sigmoid(predictions[0][1])]
            prediction = np.argmax(result, axis=-1)
            confidence = result[prediction]
            prediction_label = label[prediction]

            input_column.markdown(f"<small>Analisis selesai dalam {int(time.time() - last_time)} detik</small>", unsafe_allow_html=True)
            if prediction:
                input_column.markdown(f'<div class="error-box">Judul ini {prediction_label}.</div>', unsafe_allow_html=True)
                input_column.markdown(f'<b>Tingkat Kepercayaan:</b> {int(confidence*100)}%', unsafe_allow_html=True)
            else:
                input_column.markdown(f'<div class="success-box">Judul ini {prediction_label}.</div>', unsafe_allow_html=True)
                input_column.markdown(f'<b>Tingkat Kepercayaan:</b> {int(confidence*100)}%', unsafe_allow_html=True)
                if confidence < 0.7:
                    input_column.markdown(
                        '<div class="warning-box">Keyakinan rendah. Periksa fakta lebih lanjut dari sumber terpercaya.</div>',
                        unsafe_allow_html=True
                    )

            with st.spinner("Menghasilkan Penjelasan..."):
                explanation = query_jatevo_hoax_explanation(title, prediction_label, confidence)
                if explanation:
                    input_column.subheader("Penjelasan Generatif")
                    input_column.markdown(explanation)

            with reference_column:
                st.subheader("Artikel Referensi Terkait")
                try:
                    if "embeddings" not in data.column_names:
                        st.error("Kolom 'embeddings' tidak ditemukan.")
                    else:
                        title_embeddings = base_model.encode([title])[0]
                        similarity_score = cosine_similarity([title_embeddings], data["embeddings"]).flatten()
                        sorted_indices = np.argsort(similarity_score)[::-1].tolist()
                        if len(sorted_indices) > 0:
                            for i in sorted_indices[:5]:
                                st.markdown(
                                    f"""
                                    <small>{data['url'][i].split('/')[2] if 'url' in data.column_names else 'Sumber Tidak Tersedia'}</small>
                                    <a href="{data['url'][i] if 'url' in data.column_names else '#'}" class="reference-link">{data['title'][i]}</a>
                                    """,
                                    unsafe_allow_html=True
                                )
                        else:
                            st.warning("Tidak ada referensi yang relevan ditemukan.")
                except Exception as e:
                    st.error(f"Gagal memuat artikel referensi: {e}")
    elif submit:
        st.error("Harap masukkan URL artikel.")
except Exception as e:
    st.error(f"Terjadi kesalahan: {e}")