import streamlit as st
from utils import (
    load_and_split_documents,
    setup_vector_store,
    setup_llm,
    setup_retrieval_chain,
    format_medical_answer
)

# App title and configuration
st.set_page_config(
    page_title="Medical Information Assistant",
    page_icon="🩺",
    layout="wide"
)

# Add a sidebar
with st.sidebar:
    st.title("About")
    st.markdown("""
    ## Medical Chatbot
    This application uses RAG (Retrieval Augmented Generation) with Llama2 
    to answer medical questions based on the Gale Encyclopedia of Medicine.
    
    ### How to use:
    1. Type your medical question in the input box
    2. Press 'Ask' or hit Enter
    3. Wait for the response (it may take a few seconds)
    """)
    
    if st.button("Clear Chat History", key="clear"):
        st.session_state.messages = []
        st.rerun()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
st.title("Medical Information Assistant 🩺")
st.write("Ask your medical questions and get information from the Gale Encyclopedia of Medicine.")

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Initialize necessary components with caching
@st.cache_resource
def initialize_rag_components():
    # Load and split PDF document
    with st.spinner("Loading medical encyclopedia..."):
        documents = load_and_split_documents("The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf")
        
    # Create vector store
    db = setup_vector_store(documents[:30])
    
    # Load LLM model
    llm = setup_llm()
    
    # Create retrieval chain
    retrieval_chain = setup_retrieval_chain(db, llm)
    
    return retrieval_chain

# Initialize the RAG components
retrieval_chain = initialize_rag_components()

# Create chat input
user_input = st.chat_input("Type your medical question here...")

# Handle user input
if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message in chat
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Display assistant response in chat
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        
        try:
            # Get response from RAG chain
            response = retrieval_chain.invoke({"input": user_input})
            answer = response['answer']
            
            # Update the message placeholder with the response
            message_placeholder.markdown(answer)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            message_placeholder.markdown(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})