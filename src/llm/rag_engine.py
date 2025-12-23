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
    
    # Retrieve
    retrieved_hits = query_vector_db(question, k=7) # k=7 is a sweet spot
    if not retrieved_hits: return "No tengo información."

    context_text = format_docs(retrieved_hits)
    
    # --- DEBUG: Ver qué recuperó PDFPlumber ---
    print("\n" + "-"*20 + " DEBUG: CONTEXTO " + "-"*20)
    print(context_text[:500] + "...")
    print("-" * 60 + "\n")
    
    # --- PROMPT REFORZADO (GROUNDING) ---
    template = """
    Eres un asistente administrativo riguroso. Usa el siguiente contexto para responder.
    
    REGLAS DE ORO:
    1. Solo enumera materias si aparecen explícitamente como asignaturas en una lista o tabla (ej: "Matemática I", "Física").
    2. NO confundas los temas de una materia (ej: "Pitágoras", "Matrices") con el nombre de la materia.
    3. Si la lista no está clara, di "No encuentro la lista explícita de materias".
    
    CONTEXTO:
    {context}
    
    PREGUNTA:
    {question}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Generation
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