import streamlit as st
import time
from src import rag_engine, database

# Sayfa Temel Ayarları
st.set_page_config(
    page_title="Yerel RAG Bilgi Asistanı",
    page_icon="📚",
    layout="centered"
)

# Sade ve Özgün Arayüz İyileştirmesi
st.markdown("""
<style>
    /* Temiz Başlık ve Kart Tasarımı */
    .result-card {
        background-color: #1e293b;
        border-left: 5px solid #3b82f6;
        border-radius: 8px;
        padding: 16px;
        margin-top: 15px;
        color: #f8fafc;
    }
    .badge-offline {
        background-color: #064e3b;
        color: #6ee7b7;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- SOL MENÜ (SIDEBAR) ---
with st.sidebar:
    st.header("⚙️ Sistem Ayarları")
    top_k = st.slider("Getirilecek Parça Sayısı (Top-K):", min_value=1, max_value=5, value=3)
    
    st.markdown("---")
    st.markdown("**Çalışma Modu:** <span class='badge-offline'>Çevrim Dışı (Yerel)</span>", unsafe_allow_html=True)
    st.caption("Veriler yerel SQLite veritabanından taranır, internete çıkış yapılmaz.")
    
    st.markdown("---")
    st.subheader("📁 Yüklü Belgeler")
    docs = database.get_all_documents()
    if docs:
        # Tekrar eden isimleri engelleyip benzersiz gösterir
        unique_docs = sorted(list(set([d['filename'] for d in docs])))
        for fname in unique_docs:
            st.markdown(f"- 📄 `{fname}`")
    else:
        st.write("Veritabanında kayıtlı belge yok.")

# --- ANA EKRAN ---
st.title("📚 Yerel Belge Soru-Cevap Asistanı")
st.write("SQLite ve Vektör Benzerliği (Cosine Similarity) tabanlı yerel RAG uygulaması.")

# Soru Kutusu
user_query = st.text_input("Belgelerinizle ilgili bir soru sorun:", placeholder="Örn: Konuşma sınavları nasıl yapılacak?")

if st.button("Soruyu Yanıtla", type="primary", use_container_width=True):
    if user_query.strip():
        start_time = time.time()
        with st.spinner("SQLite veritabanı taranıyor ve analiz ediliyor..."):
            res = rag_engine.answer_query(user_query, k=top_k)
            elapsed = time.time() - start_time
            
            # Yanıt Alanı
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown("### 💬 Asistan Yanıtı:")
            st.write(res["answer"])
            st.caption(f"⏱️ Yanıt Süresi: {elapsed:.2f} saniye")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Alt Detay (Context)
            with st.expander("🔍 Veritabanından Çekilen Doküman Parçaları (Ham Bağlam)"):
                if res.get("chunks"):
                    for i, ch in enumerate(res["chunks"], 1):
                        st.markdown(f"**Parça {i}** | Belge: `{ch['filename']}` | Benzerlik Skoru: `{ch['score']:.2f}`")
                        st.info(ch['content'])
                else:
                    st.write("Eşleşen parça bulunamadı.")
    else:
        st.warning("Lütfen bir soru yazın.")