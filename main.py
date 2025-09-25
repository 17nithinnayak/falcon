from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

# LangChain components
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

# --- CONFIGURATION ---
VECTOR_STORE_PATH = "vector_store"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.1-8b-instant"

app = FastAPI(
    title="Falcon RAG API for AcadMate",
    description="A robust API to query JSSSTU course materials with relevance scores.",
    version="1.2.0"
)

# --- API Data Models ---
class Query(BaseModel):
    question: str

class SourceDocument(BaseModel):
    source: str
    page_content: str
    score: float

class Answer(BaseModel):
    answer: str
    source_documents: List[SourceDocument]

# --- Load Global Resources ---
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
    vectorstore = None
    retriever = None
    llm = None

# --- RAG Prompt and Chain ---
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
    if not vectorstore or not llm:
        raise HTTPException(status_code=503, detail="Backend resources are not available.")

    question = query.question
    
    # --- THIS IS THE FIX ---
    # 1. Get the answer by invoking the chain with just the question string.
    # The chain will handle the retrieval and context formatting internally.
    generated_answer = rag_chain.invoke(question)
    
    # 2. Separately, get the source documents with their scores for the response.
    # This ensures we have the sources without interfering with the chain's logic.
    source_docs_with_scores = vectorstore.similarity_search_with_relevance_scores(question)

    # 3. Format the source documents for the final API output.
    formatted_sources = []
    for doc, score in source_docs_with_scores:
        formatted_sources.append(
            SourceDocument(
                source=doc.metadata.get("source", "Unknown source"), 
                page_content=doc.page_content,
                score=score
            )
        )

    return {
        "answer": generated_answer,
        "source_documents": formatted_sources
    }

@app.get("/")
def read_root():
    return {"message": "Welcome to the Falcon RAG API. Send POST requests to the /ask endpoint."}

