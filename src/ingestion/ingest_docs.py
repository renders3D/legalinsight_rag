import os
import sys
import glob
from typing import List
from dotenv import load_dotenv

# --- FIX: Add project root to sys.path to allow imports from 'src' ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(project_root)

# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Custom Modules (Now this works!)
from src.utils.embeddings import get_embedding_model

# --- CONFIGURATION ---
load_dotenv(os.path.join(project_root, ".env"))

DATA_PATH = os.path.join(project_root, "data/raw_pdfs")
DB_PATH = os.getenv("CHROMA_DB_DIR", os.path.join(project_root, "vector_db"))

def load_documents() -> List[Document]:
    """
    Scans the data directory for PDF files and loads them.
    Returns a list of raw LangChain Documents.
    """
    documents = []
    pdf_files = glob.glob(os.path.join(DATA_PATH, "*.pdf"))
    
    print(f"📂 Found {len(pdf_files)} PDF(s) in {DATA_PATH}")

    for pdf_file in pdf_files:
        try:
            print(f"   [+] Loading: {os.path.basename(pdf_file)}...")
            loader = PyPDFLoader(pdf_file)
            docs = loader.load()
            documents.extend(docs)
            print(f"       -> Extracted {len(docs)} pages.")
        except Exception as e:
            print(f"   [!] Error loading {pdf_file}: {e}")

    return documents

def split_text(documents: List[Document]) -> List[Document]:
    """
    Splits raw documents into smaller semantic chunks.
    
    Mathematical Logic:
    - chunk_size=1000: Creates a semantic window.
    - chunk_overlap=200: Ensures continuity for sliding window context.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"✂️  Split {len(documents)} pages into {len(chunks)} vectorizable chunks.")
    
    if len(chunks) > 0:
        print(f"   [Sample Chunk]: {chunks[0].page_content[:100]}...")
        
    return chunks

def save_to_chroma(chunks: List[Document]):
    """
    Computes embeddings using the configured provider and saves to ChromaDB.
    """
    # 1. Get the Embedding Model (Factory Pattern)
    embeddings = get_embedding_model()

    print(f"💾 Indexing {len(chunks)} chunks into Vector Store (ChromaDB)...")
    
    try:
        # Chroma handles the batching and insertion automatically
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=DB_PATH
        )
        print(f"✅ Success! Vectors saved to '{DB_PATH}'")
        print(f"   [Note] Database uses SQLite + HNSW indices.")
        
    except Exception as e:
        print(f"❌ Error saving to ChromaDB: {e}")

def run_ingestion_pipeline():
    print("🚀 Starting Ingestion Pipeline...")
    
    # Safety Check: Warn user about mixing models
    if os.path.exists(DB_PATH):
        print(f"⚠️  Warning: '{DB_PATH}' already exists.")
        print("   If you changed the embedding model (OpenAI <-> Local), delete this folder first!\n")
    
    # Step 1: Load Raw Data
    raw_docs = load_documents()
    if not raw_docs:
        print(f"⚠️ No documents found. Add PDFs to '{DATA_PATH}'")
        return

    # Step 2: Split
    chunks = split_text(raw_docs)

    # Step 3: Embed & Store
    if chunks:
        save_to_chroma(chunks)
    else:
        print("⚠️ No text chunks created. Check PDF content.")

    print("🏁 Pipeline Finished.")

if __name__ == "__main__":
    run_ingestion_pipeline()