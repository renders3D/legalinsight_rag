import os
import sys
import glob
import re
from typing import List
from dotenv import load_dotenv

# --- PATH FIX ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(project_root)

# LangChain Imports
# FIX: Switched from PyPDFLoader to PDFPlumberLoader for better table extraction
from langchain_community.document_loaders import PDFPlumberLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.utils.embeddings import get_embedding_model

load_dotenv(os.path.join(project_root, ".env"))

DATA_PATH = os.path.join(project_root, "data/raw_pdfs")
DB_PATH = os.getenv("CHROMA_DB_DIR", os.path.join(project_root, "vector_db"))

def clean_text(text: str) -> str:
    """Standardizes text encoding artifacts."""
    if not text: return ""
    
    # Fix detached accents
    replacements = {
        r"´\s?a": "á", r"´\s?e": "é", r"´\s?i": "í", r"´\s?o": "ó", r"´\s?u": "ú",
        r"´\s?A": "Á", r"´\s?E": "É", r"´\s?I": "Í", r"´\s?O": "Ó", r"´\s?U": "Ú",
        r"~\s?n": "ñ", r"~\s?N": "Ñ",
        r"¨\s?u": "ü", r"¨\s?U": "Ü"
    }
    cleaned_text = text
    for pattern, replacement in replacements.items():
        cleaned_text = re.sub(pattern, replacement, cleaned_text)
    
    # Collapse whitespace but TRY TO PRESERVE LIST STRUCTURE
    # We replace multiple spaces with single space, but keep newlines for lists
    cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text) 
    
    return cleaned_text.strip()

def load_documents() -> List[Document]:
    documents = []
    pdf_files = glob.glob(os.path.join(DATA_PATH, "*.pdf"))
    
    print(f"📂 Found {len(pdf_files)} PDF(s) in {DATA_PATH}")

    for pdf_file in pdf_files:
        try:
            print(f"   [+] Loading (High-Res): {os.path.basename(pdf_file)}...")
            # PDFPlumber handles columns and tables better
            loader = PDFPlumberLoader(pdf_file)
            docs = loader.load()
            
            for doc in docs:
                doc.page_content = clean_text(doc.page_content)
                # Add metadata to help debugging
                doc.metadata["source"] = os.path.basename(pdf_file)
                
            documents.extend(docs)
            print(f"       -> Extracted {len(docs)} pages.")
        except Exception as e:
            print(f"   [!] Error loading {pdf_file}: {e}")

    return documents

def split_text(documents: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✂️  Split {len(documents)} pages into {len(chunks)} vectors.")
    return chunks

def save_to_chroma(chunks: List[Document]):
    embeddings = get_embedding_model()
    print(f"💾 Indexing {len(chunks)} chunks into ChromaDB...")
    try:
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=DB_PATH
        )
        print(f"✅ Success! Vectors saved to '{DB_PATH}'")
    except Exception as e:
        print(f"❌ Error saving to ChromaDB: {e}")

def run_ingestion_pipeline():
    print("🚀 Starting Ingestion Pipeline (PDFPlumber Engine)...")
    
    if os.path.exists(DB_PATH):
        print(f"♻️  Cleaning old database...")
        import shutil
        shutil.rmtree(DB_PATH)
    
    raw_docs = load_documents()
    if not raw_docs: return

    chunks = split_text(raw_docs)
    if chunks: save_to_chroma(chunks)

if __name__ == "__main__":
    run_ingestion_pipeline()