import os
import streamlit as st
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from dotenv import load_dotenv

load_dotenv()

# Streamlit UI
st.title("Sunshine Hotel & Resort Chatbot")
st.markdown("""
Welcome to Sunshine Hotel & Resort! I’m your virtual assistant, Sunny. How can I help you today?
""")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3)
    memory_length = st.slider("Memory Length", 1, 10, 5)

# Initialize OpenAI LLM
llm = ChatOpenAI(
    model="gpt-4",
    temperature=temperature,  # reduce randomness for consistency
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Initialize LangChain memory for short-term context
memory = ConversationBufferWindowMemory(
    k=memory_length,  # retain last N messages
    memory_key="history",  # Use "history" instead of "chat_history"
    return_messages=True  # for compatibility with chat models
)

vectorstore = None
retriever = None

def initialize_rag():
    global vectorstore, retriever
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory="./chroma_db"  # Local directory to store data
    )
    retriever = vectorstore.as_retriever()

initialize_rag()

# Create conversation chain
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True  # logs the prompt+response for debugging
)

# Function to extract text from uploaded files
def extract_text_from_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    elif uploaded_file.type == "text/plain":
        return uploaded_file.read().decode("utf-8")
    else:
        st.error("Unsupported file type. Please upload a PDF or TXT file.")
        return None

# Function to update the vector store with new text
def update_vector_store(text):
    if text:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(text)
        embeddings = OpenAIEmbeddings()
        global vectorstore, retriever
        vectorstore = Chroma.from_texts(
            texts=chunks,
            embedding=embeddings,
            persist_directory="./chroma_db"  # Local directory to store data
        )
        retriever = vectorstore.as_retriever()
        st.success("File uploaded and processed successfully!")

# File upload section
uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])
if uploaded_file:
    text = extract_text_from_file(uploaded_file)
    update_vector_store(text)

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add RAG context if enabled
    rag_context = ""
    if retriever:
        docs = retriever.get_relevant_documents(prompt)
        rag_context = "\n".join([d.page_content for d in docs])

    # Generate chatbot response
    response = conversation.predict(input=f"Context: {rag_context}\nUser: {prompt}")

    # Add chatbot response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)