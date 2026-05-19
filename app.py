# app.py

import os
import tempfile
from dotenv import load_dotenv

import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

# =====================================
# Load Environment Variables
# =====================================
load_dotenv()

# =====================================
# Streamlit Config
# =====================================
st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚",
    layout="wide"
)

st.title("📚 PDF RAG Chatbot")

st.markdown(
    "Upload multiple PDF files and ask questions from them."
)

# =====================================
# Session State
# =====================================
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# =====================================
# Sidebar
# =====================================
st.sidebar.header("📂 Upload PDFs")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True
)

process_btn = st.sidebar.button("⚡ Process PDFs")

# =====================================
# Process PDFs
# =====================================
if process_btn:

    if not uploaded_files:
        st.sidebar.warning("Please upload at least one PDF.")
    else:

        with st.spinner("Processing PDFs..."):

            all_docs = []

            # =========================
            # Read All PDFs
            # =========================
            for uploaded_file in uploaded_files:

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as tmp_file:

                    tmp_file.write(uploaded_file.read())
                    temp_pdf_path = tmp_file.name

                loader = PyPDFLoader(temp_pdf_path)

                docs = loader.load()

                all_docs.extend(docs)

            # =========================
            # Split Documents
            # =========================
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = splitter.split_documents(all_docs)

            # =========================
            # Embeddings
            # =========================
            embedding_model = OpenAIEmbeddings()

            # =========================
            # Create Vector DB
            # =========================
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embedding_model,
                persist_directory="Chroma-db"
            )

            st.session_state.vectorstore = vectorstore

            st.success("✅ PDFs processed successfully!")

# =====================================
# Display Chat History
# =====================================
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# =====================================
# User Input
# =====================================
query = st.chat_input(
    "Ask a question from your PDFs..."
)

# =====================================
# Question Answering
# =====================================
if query:

    if st.session_state.vectorstore is None:

        st.warning("Please upload and process PDFs first.")

    else:

        # =========================
        # Show User Message
        # =========================
        st.chat_message("user").markdown(query)

        st.session_state.messages.append({
            "role": "user",
            "content": query
        })

        # =========================
        # Retriever
        # =========================
        retriever = st.session_state.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 4,
                "fetch_k": 10,
                "lambda_mult": 0.5
            }
        )

        docs = retriever.invoke(query)

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        # =========================
        # Prompt
        # =========================
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are a helpful AI assistant.

                    Use ONLY the provided context to answer
                    the user's question.

                    If the answer is not available in the context,
                    say:
                    "I could not find the answer in the document."
                    """
                ),

                (
                    "human",
                    """
                    Context:
                    {context}

                    Question:
                    {question}
                    """
                )
            ]
        )

        final_prompt = prompt.invoke({
            "context": context,
            "question": query
        })

        # =========================
        # LLM
        # =========================
        llm = ChatMistralAI(
            model="mistral-small-2506"
        )

        # =========================
        # Generate Response
        # =========================
        with st.spinner("Thinking..."):

            response = llm.invoke(final_prompt)

            answer = response.content

        # =========================
        # Show Assistant Response
        # =========================
        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

# =====================================
# Sidebar Chat Controls
# =====================================
st.sidebar.divider()

if st.sidebar.button("🗑 Clear Chat History"):

    st.session_state.messages = []

    st.sidebar.success("Chat history cleared!")