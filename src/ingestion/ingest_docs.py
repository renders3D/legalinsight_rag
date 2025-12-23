import os
import sys
import glob
import re  # <--- Importamos Regex
from typing import List
from dotenv import load_dotenv

# --- PATH FIX ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(project_root)

# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Custom Modules
from src.utils.embeddings import get_embedding_model

# --- CONFIGURATION ---
load_dotenv(os.path.join(project_root, ".env"))

DATA_PATH = os.path.join(project_root, "data/raw_pdfs")
DB_PATH = os.getenv("CHROMA_DB_DIR", os.path.join(project_root, "vector_db"))

def clean_text(text: str) -> str:
    """
    Fixes common PDF encoding artifacts using Regex.
    Example: "Matem´ atica" -> "Matemática"
    """
    # 1. Fix detached acute accents (´ a -> á, ´a -> á)
    # The pattern looks for ´ followed by optional space, then a vowel
    replacements = {
        r"´\s?a": "á", r"´\s?e": "é", r"´\s?i": "í", r"´\s?o": "ó", r"´\s?u": "ú",
        r"´\s?A": "Á", r"´\s?E": "É", r"´\s?I": "Í", r"´\s?O": "Ó", r"´\s?U": "Ú",
        # Fix tildes (ñ) if they appear as ~ n
        r"~\s?n": "ñ", r"~\s?N": "Ñ",
        # Fix dieresis (ü)
        r"¨\s?u": "ü", r"¨\s?U": "Ü"
    }
    
    cleaned_text = text
    for pattern, replacement in replacements.items():
        cleaned_text = re.sub(pattern, replacement, cleaned_text)
    
    # 2. Fix broken hyphens usually found at line breaks (e.g. "matemá- \ntica")
    cleaned_text = re.sub(r"-\s*\n", "", cleaned_text)
    
    # 3. Collapse multiple spaces
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    
    return cleaned_text

def load_documents() -> List[Document]:
    """
    Loads PDFs and applies text cleaning.
    """
    documents = []
    pdf_files = glob.glob(os.path.join(DATA_PATH, "*.pdf"))
    
    print(f"📂 Found {len(pdf_files)} PDF(s) in {DATA_PATH}")

    for pdf_file in pdf_files:
        try:
            print(f"   [+] Loading: {os.path.basename(pdf_file)}...")
            loader = PyPDFLoader(pdf_file)
            docs = loader.load()
            
            # --- APPLY CLEANING HERE ---
            for doc in docs:
                doc.page_content = clean_text(doc.page_content)
                
            documents.extend(docs)
            print(f"       -> Extracted & Cleaned {len(docs)} pages.")
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
    print(f"✂️  Split {len(documents)} pages into {len(chunks)} vectorizable chunks.")
    
    if len(chunks) > 0:
        print(f"   [Sample Chunk]: {chunks[0].page_content[:150]}...")
        
    return chunks

def save_to_chroma(chunks: List[Document]):
    embeddings = get_embedding_model()
    
    print(f"💾 Indexing {len(chunks)} chunks into Vector Store (ChromaDB)...")
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
    print("🚀 Starting Ingestion Pipeline (With Text Cleaning)...")
    
    # Safety Check: Delete old DB to avoid mixing dirty and clean data
    if os.path.exists(DB_PATH):
        print(f"♻️  Removing old database at '{DB_PATH}' for a fresh clean index...")
        import shutil
        shutil.rmtree(DB_PATH)
    
    # Step 1: Load & Clean
    raw_docs = load_documents()
    if not raw_docs:
        print(f"⚠️ No documents found in '{DATA_PATH}'")
        return

    # Step 2: Split
    chunks = split_text(raw_docs)

    # Step 3: Embed & Store
    if chunks:
        save_to_chroma(chunks)
    else:
        print("⚠️ No text chunks created.")

    print("🏁 Pipeline Finished.")

if __name__ == "__main__":
    run_ingestion_pipeline()