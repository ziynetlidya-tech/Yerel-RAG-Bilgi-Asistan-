import os
import sys

# Hugging Face / Transformers için offline çalışma
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import pypdf
import docx

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config

from sentence_transformers import SentenceTransformer
from src import database

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

_embedding_model = None

def get_embedding_model():
    """SentenceTransformer modelini sadece yerel dosyalardan yükler."""
    global _embedding_model

    if _embedding_model is None:
        _embedding_model = SentenceTransformer(
            config.EMBEDDING_MODEL_PATH,
            local_files_only=True
        )

    return _embedding_model

def set_embedding_model(model_obj):
    """Streamlit veya dışarıdan önbelleklenmiş modeli ayarlar."""
    global _embedding_model
    _embedding_model = model_obj

def encode_texts(texts):
    """Metinleri yerel embedding modeli ile vektörlere dönüştürür."""
    model = get_embedding_model()
    return model.encode(texts, convert_to_numpy=True)

def read_file_content(file_path):
    """Farklı uzantılardaki (TXT, PDF, DOCX) dosyaların metin içeriğini okur."""
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    
    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    elif ext == ".pdf":
        reader = pypdf.PdfReader(file_path)
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    elif ext == ".docx":
        doc = docx.Document(file_path)
        for p in doc.paragraphs:
            if p.text:
                text += p.text + "\n"
    else:
        raise ValueError(f"Desteklenmeyen dosya türü: {ext}")
        
    return text.strip()

def chunk_text(text, chunk_size=500, overlap=50):
    """
    Metni ortadan bölmek yerine önce paragraflara (\n\n) göre ayırır.
    Böylece başlıklar ve maddeler birbirine karışmaz.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    
    for p in paragraphs:
        # Eğer paragraf aşırı uzun değilse doğrudan bütün bir parça olarak al
        if len(p) <= chunk_size:
            chunks.append(p)
        else:
            # Çok uzun paragrafları cümle bazlı böl
            start = 0
            while start < len(p):
                end = start + chunk_size
                chunks.append(p[start:end].strip())
                start += chunk_size - overlap
                
    return chunks if chunks else [text.strip()]

def get_embeddings(texts):
    """Metin listesinin embedding vektörlerini hesaplar."""
    embeddings = encode_texts(texts)
    return embeddings.tolist()

def ingest_text_data(filename, raw_text):
    """
    Ham metni alır, parçalar, vektörleştirir ve SQLite veritabanına kaydeder.
    """
    chunks = chunk_text(raw_text)
    if not chunks:
        return 0
        
    embeddings = get_embeddings(chunks)
    
    chunks_data = []
    for idx, (c, emb) in enumerate(zip(chunks, embeddings)):
        chunks_data.append({
            'chunk_index': idx,
            'content': c,
            'embedding': emb
        })
        
    doc_id = database.save_document_chunks(filename, chunks_data)
    print(f"[OK] '{filename}' belgesi {len(chunks)} parçaya bölünerek SQLite'a kaydedildi. (Doc ID: {doc_id})")
    return len(chunks)

def ingest_file(file_path):
    """Dosyayı okur ve veritabanına aktarır."""
    filename = os.path.basename(file_path)
    text = read_file_content(file_path)
    return ingest_text_data(filename, text)

if __name__ == "__main__":
    database.init_db()
    database.clear_db()  
    
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    
    for fname in os.listdir(data_dir):
        full_path = os.path.join(data_dir, fname)
        if os.path.isfile(full_path):
            print(f"İşleniyor: {fname}")
            ingest_file(full_path)
            
    print("Tüm belgeler tertemiz şekilde veritabanına aktarıldı!")
