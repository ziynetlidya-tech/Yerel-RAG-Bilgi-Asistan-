import os
import sys
import requests
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config
from src import database, ingestion

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Foundry Local / OpenAI Uyumlu Yerel API Uç Noktaları
FOUNDRY_LOCAL_URL = os.environ.get("FOUNDRY_LOCAL_URL", "http://localhost:5272/v1/chat/completions")
OLLAMA_LOCAL_URL = os.environ.get("OLLAMA_LOCAL_URL", "http://localhost:11434/v1/chat/completions")

def get_top_chunks(query, k=3):
    """
    Kullanıcı sorusu için SQLite veritabanından en alakalı k doküman parçasını getirir.
    (Hafta 3 Gereksinimi)
    """
    query_embeddings = ingestion.encode_texts([query])
    query_embedding = query_embeddings[0].tolist()
    
    top_chunks = database.search_similar_chunks(query_embedding, top_k=k)
    return top_chunks

def format_context_from_chunks(chunks):
    """Bulunan doküman parçalarını sistem istemine uygun metin formatına dönüştürür."""
    if not chunks:
        return ""
        
    context_str = ""
    for idx, c in enumerate(chunks, 1):
        context_str += f"[Kaynak {idx}: {c['filename']} (Parça {c['chunk_index']+1}) - Benzerlik Skoru: {c['score']:.2f}]\n"
        context_str += f"{c['content']}\n\n"
        
    return context_str.strip()

def call_local_llm_api(system_prompt, user_prompt):
    """
    Eğer bilgisayarda Foundry Local SDK veya Ollama/LocalAI REST servisi çalışıyorsa çağırır.
    Çalışmıyorsa None döndürür.
    """
    payload = {
        "model": "phi-3.5-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2
    }
    headers = {"Content-Type": "application/json"}
    
    # 1. Foundry Local Uç Noktasını Dene
    try:
        res = requests.post(FOUNDRY_LOCAL_URL, json=payload, headers=headers, timeout=5)
        if res.status_code == 200:
            data = res.json()
            return data["choices"][0]["message"]["content"]
    except Exception:
        pass
        
    # 2. Ollama / Alternatif Yerel Uç Noktayı Dene
    try:
        res = requests.post(OLLAMA_LOCAL_URL, json=payload, headers=headers, timeout=5)
        if res.status_code == 200:
            data = res.json()
            return data["choices"][0]["message"]["content"]
    except Exception:
        pass
        
    return None

def generate_fallback_grounded_answer(user_question, chunks):
    """
    Sorumlu Yapay Zeka kurallarına uygun olarak gelen bağlamı analiz eden,
    alakasız düşük skorlu parçaları filtreleyen yerel çıkarım motoru.
    """
    # Eşik değerini 0.38 yerine 0.45 yapıyoruz
    if not chunks or chunks[0]['score'] < 0.45:
        return "Üzgünüm, sağlanan belgelerde bu soruyla doğrudan ilgili bir bilgi bulunmamaktadır."
    
    # Sadece en yüksek skora çok yakın (gerçekten alakalı) parçaları al
    top_score = chunks[0]['score']
    relevant_chunks = [c for c in chunks if c['score'] >= max(0.45, top_score - 0.10)]
    
    if not relevant_chunks:
        return "Üzgünüm, sağlanan belgelerde bu soruyla doğrudan ilgili bir bilgi bulunmamaktadır."
        
    answer_parts = []
    sources = set()
    
    for c in relevant_chunks:
        # Çok kısa/kopuk parçaları filtrele (en az 40 karakter)
        if len(c['content'].strip()) > 40:
            sources.add(c['filename'])
            answer_parts.append(c['content'])
        
    if not answer_parts:
        return "Üzgünüm, sağlanan belgelerde bu soruyla doğrudan ilgili bir bilgi bulunmamaktadır."

    combined_info = "\n\n".join(answer_parts[:2])
    source_list = ", ".join(list(sources))
    
    response = f"**Yanıt (Dokümana Dayalı):** {combined_info}\n\n"
    response += f"📌 **Kaynak Gösterimi (Citation):** {source_list}"
    
    return response


def answer_query(user_question, k=3, system_prompt=config.DEFAULT_SYSTEM_PROMPT):
    chunks = get_top_chunks(user_question, k=k)
    
    if not chunks:
        return {
            "answer": "Sistemde henüz yüklenmiş bir doküman bulunmamaktadır.",
            "chunks": [],
            "sources": []
        }
        
    context_str = format_context_from_chunks(chunks)
    user_prompt = f"Aşağıdaki bağlama dayanarak soruyu yanıtla:\n\nBAĞLAM:\n{context_str}\n\nSORU: {user_question}"
    
    llm_response = call_local_llm_api(system_prompt, user_prompt)
    
    if not llm_response:
        llm_response = generate_fallback_grounded_answer(user_question, chunks)
        
    # Eşik değerini burada da 0.45 yapıyoruz
    if chunks[0]['score'] >= 0.45:
        top_score = chunks[0]['score']
        sources = list(set([c['filename'] for c in chunks if c['score'] >= max(0.45, top_score - 0.10)]))
    else:
        sources = []
    
    return {
        "answer": llm_response,
        "chunks": chunks,
        "sources": sources
    }