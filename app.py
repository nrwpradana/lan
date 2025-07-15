import streamlit as st
import re
import time
import os
import requests
import json

try:
    from src.scraper import scrape
except ImportError as e:
    st.error(f"Gagal mengimpor scraper: {e}. Pastikan modul src.scraper tersedia.")
    st.stop()

# Set page configuration
st.set_page_config(layout="wide", page_icon="🛡️", page_title="Anti Hoax Indonesia")

# Custom CSS
st.markdown("""
<style>
body { font-family: 'Arial', sans-serif; background-color: #f0f2f6; }
.stApp { max-width: 1200px; margin: 0 auto; padding: 20px; }
h1 { color: #1a73e8; text-align: center; font-size: 2.5em; margin-bottom: 10px; }
h3 { color: #333; font-size: 1.5em; margin-top: 20px; }
.stRadio > label, .stTextInput > label, .stTextArea > label { color: #444; font-weight: bold; }
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

# Load API key
try:
    API_KEY = st.secrets["JATEVO_API_KEY"]
except KeyError:
    st.error("API Key Jatevo tidak ditemukan di st.secrets. Tambahkan JATEVO_API_KEY di secrets.toml.")
    st.stop()

# Jatevo API query
def query_jatevo_fact_check(title, text=None):
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
    effective_title = title if title else "Teks tanpa judul yang jelas"
    text_portion = f"Teks (jika ada): \"{text[:600]}\"" if text else ""
    prompt = f"""
    Analisis faktual judul berikut dalam konteks Indonesia secara singkat dan formal.
    Judul: "{effective_title}"
    {text_portion}
    Ekstrak 1-2 poin utama dari judul, verifikasi kebenarannya berdasarkan fakta umum atau sumber terpercaya di Indonesia (misalnya, Kompas, data pemerintah), dan sajikan dalam format:
    - **⚠️ Headline:** [Judul]
    - **💬 Tweet Signals:** [Poin utama 1], [Poin utama 2]
    - **📰 Fact Check:** [Verifikasi poin 1], [Verifikasi poin 2]
    - **🧠 Summary:** [Ringkasan singkat]
    - **🔗 Sources:** [Sumber 1] (jika ada)
    Gunakan bahasa formal dan hindari asumsi default seperti 'Judul Tidak Tersedia'.
    """
    
    payload = {
        "model": "deepseek-ai/DeepSeek-R1-0528",
        "messages": [{"role": "user", "content": prompt}],
        "stop": [],
        "stream": False,
        "max_tokens": 200,
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
        return "Tidak ada analisis dari Jatevo API."
    except requests.exceptions.RequestException as e:
        return f"Error Jatevo API: {e}"

# UI Layout
input_column, _ = st.columns([3, 2])

with input_column:
    st.title("🛡️ Anti Hoax Indonesia")
    st.markdown("**Aplikasi deteksi hoaks berbasis AI untuk berita dalam Bahasa Indonesia.**")
    st.subheader("Masukkan Artikel")
    input_type = st.radio("Pilih jenis input:", ("URL Artikel", "Teks Langsung"))
    
    if input_type == "URL Artikel":
        user_input = st.text_input("URL Artikel", placeholder="https://example.com/berita", help="Masukkan URL artikel berita dalam Bahasa Indonesia.")
    else:
        user_input = st.text_area("Teks Artikel", placeholder="Masukkan teks artikel...", height=150)
    
    submit = st.button("Cek Hoaks")

try:
    if submit and user_input:
        last_time = time.time()
        text = user_input
        title = None

        if input_type == "URL Artikel":
            with st.spinner("Membaca Artikel..."):
                try:
                    scrape_result = scrape(user_input)
                    title, text = scrape_result.title, scrape_result.text
                except Exception as e:
                    st.error(f"Gagal mengambil data artikel dari URL: {e}")
                    st.stop()
        else:  # Teks Langsung
            # Gunakan baris pertama sebagai proksi judul jika ada, atau teks utuh
            lines = text.split("\n")
            title = lines[0].strip() if lines and lines[0].strip() else text[:50]  # Ambil 50 karakter pertama jika tidak ada baris

        if title:
            with st.spinner("Menganalisis Hoaks..."):
                analysis = query_jatevo_fact_check(title, text if text != title else None)
                input_column.markdown(
                    f"<small>Analisis selesai dalam {int(time.time() - last_time)} detik</small>",
                    unsafe_allow_html=True,
                )
                input_column.markdown(analysis, unsafe_allow_html=True)
    elif submit:
        st.error("Harap masukkan URL atau teks artikel.")
except Exception as e:
    st.error(f"Terjadi kesalahan: {e}")