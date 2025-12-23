import os
import sys
import argparse
from dotenv import load_dotenv

# --- PATH FIX ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(project_root)

# LangChain Imports
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Local Imports
from src.retrieval.search import query_vector_db

# --- CONFIGURATION ---
load_dotenv(os.path.join(project_root, ".env"))
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()

def get_llm():
    """Factory to initialize the LLM based on configuration."""
    if LLM_PROVIDER == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key: raise ValueError("OPENAI_API_KEY missing.")
        print("🤖 Initializing LLM: OpenAI GPT-3.5-turbo")
        return ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    elif LLM_PROVIDER == "ollama":
        print("🦙 Initializing LLM: Local Llama 3 via Ollama")
        return ChatOllama(model="llama3", temperature=0)
    
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")

def format_docs(docs_with_scores):
    return "\n\n".join([doc.page_content for doc, score in docs_with_scores])

def run_rag_pipeline(question: str):
    print(f"\n💬 Question: {question}")
    
    # 1. Retrieval
    # OPTIMIZATION: Increased k from 5 to 10 to capture lists spread across chunks
    retrieved_hits = query_vector_db(question, k=10)
    
    if not retrieved_hits:
        return "No tengo información suficiente en los documentos."

    context_text = format_docs(retrieved_hits)
    
    # --- DEBUGGING ---
    print("\n" + "-"*20 + " DEBUG: CONTEXTO (Primeros 500 chars) " + "-"*20)
    print(context_text[:500] + "...\n(truncated)")
    print("-" * 60 + "\n")
    # -----------------
    
    # 2. Augmentation
    template = """
    Eres un asistente académico experto. Usa EXCLUSIVAMENTE el siguiente contexto para responder la pregunta.
    
    Instrucciones:
    1. Si encuentras una lista de materias, enuméralas.
    2. Si el texto está cortado, intenta inferir el contexto lógico.
    3. Si de verdad no hay respuesta, di "No encuentro esa información".
    
    CONTEXTO:
    {context}
    
    PREGUNTA:
    {question}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # 3. Generation
    llm = get_llm()
    rag_chain = prompt | llm | StrOutputParser()
    
    print("🧠 Generating Answer...")
    response = rag_chain.invoke({
        "context": context_text,
        "question": question
    })
    
    return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LegalInsight RAG Chat")
    parser.add_argument("query", type=str, nargs="?", help="Question", default="¿Cuáles son las materias obligatorias?")
    args = parser.parse_args()

    answer = run_rag_pipeline(args.query)
    
    print("\n" + "="*50)
    print(f"📝 RAG Answer:\n{answer}")
    print("="*50 + "\n")