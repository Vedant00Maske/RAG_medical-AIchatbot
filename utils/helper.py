import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict, Any, Tuple

def load_and_split_documents(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 20) -> List:
    """
    Load PDF document and split it into chunks.
    Args:
        pdf_path: Path to the PDF file.
        chunk_size: Size of each text chunk.
        chunk_overlap: Overlap between chunks.
    Returns:
        List of document chunks.
    """
    try:
        # Load the PDF using LangChain's PyPDFLoader
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        
        # Split the loaded documents into smaller chunks for processing
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        documents = text_splitter.split_documents(docs)
        return documents
    except Exception as e:
        # Display error in Streamlit UI if loading fails
        st.error(f"Error loading document: {str(e)}")
        return []

def setup_vector_store(documents: List, model_name: str = "llama2", temperature: float = 0.35) -> Any:
    """
    Create a vector store from documents using Ollama embeddings.
    Args:
        documents: List of document chunks.
        model_name: Name of the embedding model.
        temperature: Temperature parameter for embeddings.
    Returns:
        FAISS vector store.
    """
    try:
        # Create embeddings for the documents using Ollama
        embeddings = OllamaEmbeddings(model=model_name, temperature=temperature)
        # Store the embeddings in a FAISS vector store for similarity search
        db = FAISS.from_documents(documents, embeddings)
        return db
    except Exception as e:
        # Display error in Streamlit UI if vector store creation fails
        st.error(f"Error setting up vector store: {str(e)}")
        return None

def setup_llm(model_name: str = "llama2") -> Any:
    """
    Initialize the LLM model.
    Args:
        model_name: Name of the model to use.
    Returns:
        Initialized LLM.
    """
    try:
        # Initialize the Ollama LLM with the specified model name
        return Ollama(model=model_name)
    except Exception as e:
        # Display error in Streamlit UI if LLM setup fails
        st.error(f"Error setting up LLM: {str(e)}")
        return None

def create_medical_prompt() -> ChatPromptTemplate:
    """
    Create a chat prompt template for medical questions.
    Returns:
        ChatPromptTemplate for medical questions.
    """
    # Define a prompt template to instruct the LLM on how to answer
    return ChatPromptTemplate.from_template("""
    Answer the following medical question based only on the provided context.
    Think step by step before providing a detailed answer.
    If you do not find any relevant context, please say "I don't have enough information to answer that question."
    Focus on providing accurate medical information in a friendly, helpful tone.
    Format any medical terms or important information in **bold**.
    
    <context>
    {context}
    </context>
    
    Question: {input}
    """)

def setup_retrieval_chain(db: Any, llm: Any) -> Any:
    """
    Set up the retrieval chain for answering questions.
    Args:
        db: Vector store database.
        llm: Language model.
    Returns:
        Retrieval chain.
    """
    try:
        # Create the prompt template for the LLM
        prompt = create_medical_prompt()
        
        # Create a document chain that combines the LLM and prompt
        document_chain = create_stuff_documents_chain(llm, prompt)
        
        # Convert the vector store into a retriever for document search
        retriever = db.as_retriever()
        
        # Create a retrieval chain that uses the retriever and document chain
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        return retrieval_chain
    except Exception as e:
        # Display error in Streamlit UI if retrieval chain setup fails
        st.error(f"Error setting up retrieval chain: {str(e)}")
        return None

def format_medical_answer(answer: str) -> str:
    """
    Format medical answer for better readability.
    Args:
        answer: Raw answer from the model.
    Returns:
        Formatted answer.
    """
    # Add any additional formatting logic here if needed
    return answer

def get_sources_from_response(response: Dict[str, Any]) -> List[str]:
    """
    Extract source documents from the response.
    Args:
        response: Response from the retrieval chain.
    Returns:
        List of source document references.
    """
    sources = []
    # Check if context is present and is a list of documents
    if 'context' in response and isinstance(response['context'], list):
        for doc in response['context']:
            # Extract the 'source' metadata if available
            if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                source = doc.metadata['source']
                if source not in sources:
                    sources.append(source)
    
    return sources