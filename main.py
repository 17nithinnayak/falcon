from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

# LangChain components for building the RAG pipeline
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

# Load the API key from the .env file
load_dotenv()

# --- CONFIGURATION ---
VECTOR_STORE_PATH = "vector_store"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.1-8b-instant"

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Falcon RAG API for AcadMate",
    description="A robust API to query JSSSTU course materials.",
    version="1.0.0"
)

# --- API Data Models (for request/response validation) ---
class Query(BaseModel):
    question: str

class SourceDocument(BaseModel):
    source: str
    page_content: str

class Answer(BaseModel):
    answer: str
    source_documents: List[SourceDocument]

# --- Load Global Resources (Done once on startup) ---
try:
    print("Loading embedding model and vector store...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(persist_directory=VECTOR_STORE_PATH, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4}) 
    
    print("Initializing Groq LLM...")
    llm = ChatGroq(model_name=LLM_MODEL)
    print("All resources loaded successfully!")
except Exception as e:
    print(f"FATAL: Error initializing backend resources: {e}")
    retriever = None
    llm = None

# --- RAG Prompt Template ---
template = """
You are Falcon, an expert AI teaching assistant for students at JSS Science and Technology University.
Your mission is to help students understand concepts clearly by providing answers based ONLY on the context provided from their official course materials.
- Do not use any information outside of the provided context.
- If the answer is not found in the context, clearly state that the information is not available in the provided documents.
- Be friendly, encouraging, and break down complex topics into simple, easy-to-understand steps.
- Structure your answers clearly, using bullet points or numbered lists if it helps with clarity.

Context from the textbook:
{context}

Student's Question:
{question}

Your Helpful Answer:
"""
prompt = PromptTemplate.from_template(template)

# --- RAG Chain Definition ---
def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- API Endpoint ---
@app.post("/ask", response_model=Answer)
async def ask_question(query: Query):
    """
    Accepts a student's question, retrieves context, and generates a helpful answer.
    """
    if not retriever or not llm:
        raise HTTPException(status_code=503, detail="Backend resources are not available.")

    question = query.question
    
    # Retrieve source documents. We use invoke() now instead of the deprecated method.
    source_docs = retriever.invoke(question)
    
    # Invoke the RAG chain to generate the final answer
    generated_answer = rag_chain.invoke(question)

    # --- THIS IS THE FIX ---
    # Manually format the source documents to match our Pydantic model.
    # We extract the 'source' from the metadata of each document.
    formatted_sources = []
    for doc in source_docs:
        formatted_sources.append(
            SourceDocument(
                source=doc.metadata.get("source", "Unknown source"), 
                page_content=doc.page_content
            )
        )

    return {
        "answer": generated_answer,
        "source_documents": formatted_sources # Return the correctly formatted list
    }

@app.get("/")
def read_root():
    return {"message": "Welcome to the Falcon RAG API. Send POST requests to the /ask endpoint."}

