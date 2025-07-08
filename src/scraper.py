# src/scraper.py

import requests
from newspaper import Article, Config
import time
from dataclasses import dataclass
from typing import Optional

@dataclass
class ScrapeResult:
    """Class untuk menyimpan hasil scraping artikel."""
    title: str
    text: str
    url: str
    success: bool = False

class Scraper:
    """Kelas untuk scraping artikel berita dengan konfigurasi yang dapat disesuaikan."""
    
    def __init__(self):
        # Konfigurasi default untuk newspaper3k
        self.config = Config()
        self.config.browser_user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        self.config.request_timeout = 10  # Timeout dalam detik
        self.config.memoize_articles = True  # Aktifkan caching internal

    def scrape(self, url: str) -> Optional[ScrapeResult]:
        """
        Scrap artikel berita dari URL yang diberikan.

        Args:
            url (str): URL artikel berita yang akan di-scrape.

        Returns:
            Optional[ScrapeResult]: Objek dengan title, text, url, dan status success.
                                   Mengembalikan None jika gagal.

        Raises:
            ValueError: Jika URL tidak valid.
        """
        if not url or not isinstance(url, str):
            raise ValueError("URL harus berupa string yang valid.")

        try:
            # Inisialisasi objek Article dengan konfigurasi
            article = Article(url, language="id", config=self.config)
            
            # Unduh dan parse artikel
            start_time = time.time()
            article.download()
            article.parse()
            
            # Periksa apakah teks berhasil diekstrak
            if not article.text or not article.text.strip():
                print(f"Warning: Tidak ada teks yang diekstrak dari {url}")
                return ScrapeResult(title="Judul Tidak Ditemukan", text="Teks Tidak Ditemukan", url=url, success=False)
            
            # Hitung waktu eksekusi
            elapsed_time = time.time() - start_time
            print(f"Scraping {url} selesai dalam {elapsed_time:.2f} detik")

            # Kembalikan hasil
            return ScrapeResult(
                title=article.title or "Judul Tidak Ditemukan",
                text=article.text,
                url=url,
                success=True
            )

        except requests.RequestException as e:
            print(f"Error jaringan saat scraping {url}: {e}")
            return ScrapeResult(title="Judul Tidak Ditemukan", text="Teks Tidak Ditemukan", url=url, success=False)
        except Exception as e:
            print(f"Error saat parsing {url}: {e}")
            return ScrapeResult(title="Judul Tidak Ditemukan", text="Teks Tidak Ditemukan", url=url, success=False)

# Instansiasi scraper untuk penggunaan langsung
scraper = Scraper()

if __name__ == "__main__":
    # Contoh penggunaan
    test_urls = [
        "https://www.kompas.com/read/2023/01/01/1234567/judul-berita",
        "https://invalid-url-example.com"  # URL tidak valid untuk pengujian error
    ]
    for url in test_urls:
        result = scraper.scrape(url)
        if result:
            print(f"URL: {result.url}")
            print(f"Judul: {result.title}")
            print(f"Teks: {result.text[:200]}...")  # Tampilkan 200 karakter pertama
            print(f"Sukses: {result.success}\n")
        else:
            print("Gagal mengambil artikel.\n")