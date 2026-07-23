import os
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.vector_stores.types import VectorStoreQuery

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Step 2: Set up embedding model globally
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 3: Load the PDF
documents = SimpleDirectoryReader(input_files=["sample.pdf"]).load_data()

print(f"Loaded {len(documents)} document(s)")
print(documents[0].text[:200])

# Step 4: Set up pgvector store, and only build index if data doesn't already exist
vector_store = PGVectorStore.from_params(
    database="tododb",
    host="localhost",
    password="mysecret",
    port=5432,
    user="postgres",
    table_name="llamaindex_week2_chunks",
    embed_dim=384,
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Check if data already exists using a direct query on the vector store
existing_query_result = vector_store.query(
    __import__("llama_index.core.vector_stores.types", fromlist=["VectorStoreQuery"]).VectorStoreQuery(
        query_embedding=Settings.embed_model.get_query_embedding("test"),
        similarity_top_k=1,
    )
)

if len(existing_query_result.nodes) == 0:
    print("No existing data found. Building index for the first time...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )
    print("Documents successfully indexed and stored in pgvector!")
else:
    print("Data already exists in pgvector. Loading existing index instead of re-inserting.")
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
    )

# Step 5: Retrieve top-3 relevant chunks for a query
query = "Why does RAG reduce hallucination?"

retriever = index.as_retriever(similarity_top_k=3)
retrieved_nodes = retriever.retrieve(query)

print(f"\nQuery: {query}\n")
for i, node in enumerate(retrieved_nodes, start=1):
    print(f"--- Chunk {i} ---")
    print(node.text)
    print()    

# Step 6: Generate an answer using Gemini, based on retrieved chunks
llm = GoogleGenAI(
    model="gemini-flash-latest",
    api_key=GOOGLE_API_KEY
)

context = "\n\n".join([node.text for node in retrieved_nodes])

prompt = f"""Answer the question based only on the following context.
If the answer is not in the context, say you don't know.

Context:
{context}

Question: {query}

Answer:"""

response = llm.complete(prompt)

print("=== Final Generated Answer ===")
print(response.text)    