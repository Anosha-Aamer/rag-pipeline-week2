# RAG Pipeline — Week 2 (LangChain vs LlamaIndex)

This project implements a basic Retrieval-Augmented Generation (RAG) pipeline using two different frameworks — **LangChain** and **LlamaIndex** — to compare their approach to building RAG systems.

## What is RAG?

RAG (Retrieval-Augmented Generation) combines information retrieval with text generation to reduce hallucination and give LLMs access to external, up-to-date knowledge. Instead of relying only on what the model learned during training, RAG retrieves relevant chunks of text from a knowledge base and passes them to the LLM as context before generating an answer.

## Pipeline Overview

Both pipelines follow the same core steps:

1. **Ingestion** — Load a sample PDF
2. **Chunking** — Split the document into smaller pieces
3. **Embedding** — Convert chunks into vectors using `all-MiniLM-L6-v2` (384 dimensions)
4. **Vector Storage** — Store embeddings in **pgvector** (PostgreSQL extension)
5. **Retrieval** — Retrieve top-3 relevant chunks for a given query
6. **Generation** — Pass retrieved chunks + query to **Gemini (gemini-flash-latest)** to generate a grounded answer

## Tech Stack

- **LangChain** / **LlamaIndex** — RAG frameworks
- **pgvector** — vector similarity search inside PostgreSQL (run via Docker)
- **sentence-transformers (all-MiniLM-L6-v2)** — embedding model
- **Google Gemini API** — LLM for answer generation
- **Python 3.14**

## Setup

1. Clone the repo and create a virtual environment:
```bash
   python -m venv venv
   venv\Scripts\Activate   # Windows
```

2. Install dependencies:
```bash
   pip install -r requirements.txt
```

3. Create a `.env` file with your Gemini API key:
   GOOGLE_API_KEY=your_api_key_here

4. Start pgvector via Docker (Postgres + pgvector extension enabled):
```bash
   docker compose up -d
```

5. Run either pipeline:
```bash
   python langchain_pipeline.py
   python llamaindex_pipeline.py
```

## LangChain vs LlamaIndex — Comparison

| Aspect | LangChain | LlamaIndex |
|---|---|---|
| Design focus | General-purpose LLM app framework | RAG/data-indexing focused |
| Pipeline steps | Explicit — loader, splitter, embeddings, vector store all separate | More automated — `VectorStoreIndex.from_documents()` handles chunking + embedding + storage in one call |
| Chunking control | Explicit chunk size/overlap (`RecursiveCharacterTextSplitter`) | Uses its own default splitting strategy unless configured |
| Retrieval | `vectorstore.similarity_search()` | `index.as_retriever()` |
| Generation call | `llm.invoke(prompt)` | `llm.complete(prompt)` |
| Chunk content attribute | `doc.page_content` | `node.text` |
| Observed difference | Split sample PDF into 4 distinct chunks | Split into fewer, larger chunks — retrieval returned only 2 chunks covering the same content |

Both approaches produced accurate, grounded answers for the test query: *"Why does RAG reduce hallucination?"*

## Files

- `langchain_pipeline.py` — RAG pipeline built with LangChain
- `llamaindex_pipeline.py` — RAG pipeline built with LlamaIndex
- `docker-compose.yml` — pgvector (PostgreSQL) setup via Docker
- `sample.pdf` — sample document used for testing
- `requirements.txt` — all dependencies

---

**Author:** Anosha Aamer

**Internship:** Venturenox — ML/Data Analysis Track

**Week 2 Deliverable:** RAG Theory & First Pipeline (LangChain + LlamaIndex, pgvector, Gemini)
