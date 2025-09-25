import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import time

# --- CONFIGURATION ---
DATA_PATH = "data/"
VECTOR_STORE_PATH = "vector_store"
EMBEDDING_MODEL = "all-MiniLM-L6-v2" # A powerful and popular open-source model
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

def ingest_data():
    """
    Reads all PDF documents from the data path, splits them into chunks,
    creates embeddings, and stores them in a persistent ChromaDB vector store.
    This is the "heavy lifting" part that is done offline.
    """
    start_time = time.time()
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data directory '{DATA_PATH}' not found.")
        return

    pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"No PDF files found in '{DATA_PATH}'. Please add your textbooks to this folder.")
        return

    print(f"--- Starting Data Ingestion ---")
    print(f"Found {len(pdf_files)} PDF(s) to process: {pdf_files}")

    all_docs = []
    for pdf_file in pdf_files:
        file_path = os.path.join(DATA_PATH, pdf_file)
        print(f"Loading document: {file_path}...")
        try:
            loader = PyMuPDFLoader(file_path)
            documents = loader.load()
            all_docs.extend(documents)
        except Exception as e:
            print(f"Error loading {pdf_file}: {e}")
            continue

    if not all_docs:
        print("No documents were successfully loaded. Aborting.")
        return
        
    print(f"Successfully loaded {len(all_docs)} pages from all PDFs.")

    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    split_docs = text_splitter.split_documents(all_docs)
    print(f"Split documents into {len(split_docs)} chunks.")

    print(f"Initializing embedding model: {EMBEDDING_MODEL}...")
    # This will download the model from Hugging Face on the first run
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print(f"Creating and persisting vector store at '{VECTOR_STORE_PATH}'...")
    print("This may take a long time depending on the number and size of your documents...")
    
    # Create the vector store and compute embeddings. This is the most time-consuming step.
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=VECTOR_STORE_PATH
    )
    
    end_time = time.time()
    print("\n--- Ingestion Complete ---")
    print(f"Vector store created successfully with {len(split_docs)} chunks.")
    print(f"Total time taken: {end_time - start_time:.2f} seconds.")
    print("You can now run the main.py API server to start asking questions.")

if __name__ == "__main__":
    ingest_data()

