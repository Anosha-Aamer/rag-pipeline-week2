import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env file
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Step 3: Load the PDF
loader = PyPDFLoader("sample.pdf")
documents = loader.load()

print(f"Loaded {len(documents)} page(s) from PDF")
print(documents[0].page_content[:200])  # preview first 200 characters

# Step 4: Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)

print(f"Split into {len(chunks)} chunks")
print(chunks[0].page_content)

# Step 5: Set up the embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Quick test: embed the first chunk and check vector size
test_vector = embeddings.embed_query(chunks[0].page_content)
print(f"Embedding vector length: {len(test_vector)}")
print(test_vector[:5])  # preview first 5 numbers

# Step 6: Connect to pgvector, and only insert chunks if collection is empty
CONNECTION_STRING = "postgresql+psycopg://postgres:mysecret@localhost:5432/tododb"
COLLECTION_NAME = "rag_week2_chunks"

# First, connect to (or create) the vectorstore WITHOUT inserting anything yet
vectorstore = PGVector(
    embeddings=embeddings,
    collection_name=COLLECTION_NAME,
    connection=CONNECTION_STRING,
    use_jsonb=True,
)

# Check how many chunks are already stored
existing_docs = vectorstore.similarity_search("", k=1)  # quick check

if len(existing_docs) == 0:
    print("No existing data found. Inserting chunks for the first time...")
    vectorstore.add_documents(chunks)
    print("Chunks successfully stored in pgvector!")
else:
    print("Data already exists in pgvector. Skipping insertion.")

# Step 7: Retrieve top-3 relevant chunks for a query
query = "Why does RAG reduce hallucination?"

retrieved_docs = vectorstore.similarity_search(query, k=3)

print(f"\nQuery: {query}\n")
for i, doc in enumerate(retrieved_docs, start=1):
    print(f"--- Chunk {i} ---")
    print(doc.page_content)
    print()

# Step 8: Generate an answer using Gemini, based on retrieved chunks
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    google_api_key=GOOGLE_API_KEY
)

# Combine retrieved chunks into one context block
context = "\n\n".join([doc.page_content for doc in retrieved_docs])

prompt = f"""Answer the question based only on the following context. 
If the answer is not in the context, say you don't know.

Context:
{context}

Question: {query}

Answer:"""

response = llm.invoke(prompt)

print("=== Final Generated Answer ===")
if isinstance(response.content, list):
    final_answer = response.content[0]["text"]
else:
    final_answer = response.content

print(final_answer) 