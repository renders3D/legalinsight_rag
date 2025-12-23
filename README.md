# ⚖️ Legal Insight AI: RAG Engine

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-Orchestration-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-orange)

**LegalInsight AI** is a Retrieval-Augmented Generation (RAG) system designed to ingest technical legal documentation, index it into a vector space, and answer natural language queries with precise citations.

## 🧠 Core Concepts

1.  **Ingestion:** Loading PDF documents and splitting text into semantic chunks.
2.  **Embedding:** Converting text chunks into vectors ($\mathbb{R}^n$) using Transformer models.
3.  **Storage:** Indexing vectors in **ChromaDB** for efficient similarity search.
4.  **Retrieval:** Finding the top-$k$ most relevant chunks for a user query (Cosine Similarity).
5.  **Generation:** Using an LLM (Large Language Model) to synthesize an answer based *only* on the retrieved context.

## 🛠️ Tech Stack

* **Orchestrator:** LangChain.
* **Vector DB:** ChromaDB (Local).
* **Embeddings:** Hugging Face (Sentence Transformers) or OpenAI.
* **LLM:** OpenAI GPT-4o or Local Llama 3 via Ollama.

## 📂 Structure

```text
src/
├── ingestion/   # ETL Pipeline: PDF -> Vectors
├── retrieval/   # Search Logic
└── llm/         # Generator Logic
```

##
*Project Manager: Carlos Luis Noriega*