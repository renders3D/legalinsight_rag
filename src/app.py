import streamlit as st
import os
import sys
import shutil
import time

# --- PATH FIX ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.append(project_root)

# Importar nuestros módulos (Tu código matemático)
from src.ingestion.ingest_docs import run_ingestion_pipeline
from src.llm.rag_engine import run_rag_pipeline

# --- CONFIG ---
st.set_page_config(page_title="LegalInsight AI", page_icon="⚖️", layout="wide")
DATA_PATH = os.path.join(project_root, "data/raw_pdfs")

# --- UI STYLES ---
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .stChatMessage[data-testid="stChatMessageUser"] {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Mensaje de bienvenida del bot
    st.session_state.messages.append({
        "role": "assistant", 
        "content": "Hola. Soy tu asistente legal/técnico. Sube un documento PDF para comenzar a analizarlo."
    })

def save_uploaded_file(uploaded_file):
    """Guarda el archivo subido en la carpeta de datos y dispara la ingestión."""
    # 1. Limpiar carpeta de datos anterior para evitar mezclas
    if os.path.exists(DATA_PATH):
        shutil.rmtree(DATA_PATH)
    os.makedirs(DATA_PATH, exist_ok=True)
    
    # 2. Guardar nuevo archivo
    file_path = os.path.join(DATA_PATH, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path

# --- SIDEBAR (INGESTION) ---
with st.sidebar:
    st.header("📂 Document Center")
    uploaded_file = st.file_uploader("Subir PDF Legal/Técnico", type=["pdf"])
    
    if uploaded_file:
        if st.button("Procesar Documento", type="primary"):
            with st.spinner("⚙️ Ingestando documento... (Limpieza -> Chunking -> Vectores)"):
                # Guardar
                save_uploaded_file(uploaded_file)
                
                # Ejecutar tu pipeline de ingestión (el que arreglamos con Regex)
                # Redirigimos stdout para capturar los logs en la UI si quisiéramos, 
                # pero por ahora solo corremos la función.
                run_ingestion_pipeline()
                
                st.success("✅ ¡Indexación completada!")
                st.session_state.messages = [] # Reset chat on new doc
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": f"He leído '{uploaded_file.name}'. ¿Qué deseas saber sobre este documento?"
                })
                time.sleep(1)
                st.rerun()

    st.divider()
    st.info("Modelo Vectorial: all-MiniLM-L6-v2 (R^384)")
    st.info("LLM: Llama 3 (Local) / GPT-3.5")

# --- MAIN CHAT INTERFACE ---
st.title("⚖️ LegalInsight AI")
st.caption("RAG Engine powered by LangChain & ChromaDB")

# 1. Renderizar historial
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 2. Input de usuario
if prompt := st.chat_input("Escribe tu pregunta jurídica o técnica..."):
    # Guardar y mostrar mensaje de usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 3. Generar respuesta (RAG)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("🧠 Consultando jurisprudencia (Búsqueda Vectorial)..."):
            try:
                # Llamada al motor RAG
                response_text = run_rag_pipeline(prompt)
                
                # Simular efecto de escritura (streaming simulado)
                message_placeholder.markdown(response_text)
                full_response = response_text
                
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                message_placeholder.error(error_msg)
                full_response = error_msg
        
        # Guardar respuesta en historial
        st.session_state.messages.append({"role": "assistant", "content": full_response})