import os
import sys
import argparse
from typing import List, Tuple
from dotenv import load_dotenv

# --- PATH FIX: Add project root to sys.path ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(project_root)

from langchain_chroma import Chroma
from langchain_core.documents import Document
from src.utils.embeddings import get_embedding_model

# --- CONFIGURATION ---
load_dotenv(os.path.join(project_root, ".env"))
DB_PATH = os.getenv("CHROMA_DB_DIR", os.path.join(project_root, "vector_db"))

def query_vector_db(query: str, k: int = 3) -> List[Tuple[Document, float]]:
    """
    Searches the Vector Database for the k most similar chunks to the query.
    
    Args:
        query (str): The natural language question.
        k (int): Number of results to retrieve.
        
    Returns:
        List of (Document, score). 
        Note: In Chroma, lower score means closer distance (better match) if using L2.
    """
    print(f"🔍 Querying Vector Space: '{query}'")
    
    # 1. Initialize Embedding Function (Must match ingestion model!)
    embedding_function = get_embedding_model()
    
    # 2. Load Existing Database
    if not os.path.exists(DB_PATH):
        print(f"❌ Error: Database not found at {DB_PATH}. Run ingestion first.")
        return []

    db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embedding_function
    )
    
    # 3. Perform Similarity Search
    # similarity_search_with_score returns the chunk and the distance metric
    results = db.similarity_search_with_score(query, k=k)
    
    return results

if __name__ == "__main__":
    # Allow running from command line with arguments
    parser = argparse.ArgumentParser(description="Query the LegalInsight Vector DB")
    parser.add_argument("query", type=str, nargs="?", help="The question to ask", default="Plan de estudios")
    args = parser.parse_args()

    # Run Search
    hits = query_vector_db(args.query, k=3)
    
    print(f"\n📊 Search Results (Top {len(hits)}):")
    print("-" * 50)
    
    for i, (doc, score) in enumerate(hits):
        # Metadata usually contains page number and source file
        source = os.path.basename(doc.metadata.get("source", "Unknown"))
        page = doc.metadata.get("page", "N/A")
        
        print(f"📄 Result #{i+1} | Distance: {score:.4f} | Source: {source} (Pg {page})")
        print(f"💬 Content Snippet: {doc.page_content[:200].replace(chr(10), ' ')}...")
        print("-" * 50)