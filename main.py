# -*- coding: utf-8 -*-
"""
Created on 06 Oct 2025
@author: Jaju Peter
"""

import os
import time
import streamlit as st
from dotenv import load_dotenv
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS  # ✅ FAISS for local vector storage

# ==========================================================
# 🎨 PAGE CONFIGURATION
# ==========================================================
st.set_page_config(
    page_title="InsightBot: Research Tool",
    page_icon="📈",
    layout="wide",
)

# ==========================================================
# CUSTOM STYLING
# ==========================================================
st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
        color: #e6e6e6;
        font-family: 'Inter', sans-serif;
    }
    h1 {
        color: #93c5fd;
        text-align: center;
        font-weight: 700;
    }
    .stButton>button {
        background-color: #2563EB;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #3B82F6;
        color: #f0f0f0;
    }
    .description {
        background-color: #1e293b;
        padding: 1.2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        color: #dbeafe;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

openai_api_key = st.sidebar.text_input('OpenAI API Key')
# ==========================================================
# 🔑 LOAD OPENAI API KEY (FROM STREAMLIT SECRETS OR .ENV)
# ==========================================================
#load_dotenv()  # Load local .env if available

# First, try Streamlit secrets (Cloud deployment)
#openai_api_key = st.secrets.get("OPENAI_API_KEY") if "OPENAI_API_KEY" in st.secrets else None

# Fallback to environment variable (.env or local)
#if not openai_api_key:
    #openai_api_key = os.getenv("OPENAI_API_KEY")

# Stop app if key is missing
#if not openai_api_key:
   # st.error("❌ Missing OpenAI API key. Please add it in Streamlit Secrets or your .env file.")
    #st.stop()

# Set environment variable
#os.environ["OPENAI_API_KEY"] = openai_api_key

# ==========================================================
# 🧠 APP HEADER & DESCRIPTION
# ==========================================================
st.title("📈 InsightBot: Research Tool")

st.markdown(
    """
    <div class="description">
    <p>
    Welcome to <b>InsightBot</b> — your AI-powered research companion that transforms information overload into instant understanding.
    <br><br>
    Simply share up to three article URLs, and InsightBot will:
    <ul>
        <li>Read, analyze, and extract key insights from each article in seconds 🧠</li>
        <li>Map connections and uncover deeper meaning using advanced AI embeddings ⚙️ </li>
        <li>Deliver clear, concise answers backed by verified sources and intelligent reasoning 🔍</li>
    </ul>
    Whether you’re conducting academic research, analyzing market trends, or just staying ahead of the news cycle — InsightBot turns data into knowledge and knowledge into action ⚡ 
    </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==========================================================
# 📰 SIDEBAR URL INPUT
# ==========================================================
st.sidebar.header("📎 News Article URLs")
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}", key=f"url_{i}")
    if url:
        urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

# ==========================================================
# 🧱 FAISS STORAGE PATH
# ==========================================================
index_path = "faiss_index"

# ==========================================================
# 🧠 MAIN LOGIC
# ==========================================================
main_placeholder = st.empty()

if process_url_clicked:
    if not urls:
        st.warning("⚠️ Please enter at least one valid URL.")
    else:
        with st.spinner("🔍 Fetching and processing data... Please wait."):
            loader = UnstructuredURLLoader(urls=urls)
            data = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=1000
            )
            docs = text_splitter.split_documents(data)

            embeddings = OpenAIEmbeddings()
            vectorstore_openai = FAISS.from_documents(docs, embeddings)
            time.sleep(1.5)

            vectorstore_openai.save_local(index_path)
            st.success("✅ Articles processed and stored successfully!")

# ==========================================================
# 💬 QUERY SECTION
# ==========================================================
query = st.text_input("🔎 Ask a question about the articles:")
if query:
    if not os.path.exists(index_path):
        st.error("❌ No FAISS index found. Please process URLs first.")
    else:
        with st.spinner("🤔 Thinking..."):
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.load_local(
                index_path,
                embeddings,
                allow_dangerous_deserialization=True
            )
            llm = OpenAI(temperature=0.7, max_tokens=500)
            chain = RetrievalQAWithSourcesChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever()
            )
            result = chain({"question": query}, return_only_outputs=True)
        st.success("✅ Done!")

        st.header("📘 Answer")
        st.write(result["answer"])

        sources = result.get("sources", "")
        if sources:
            source_links = list({s.strip().strip(',') for s in sources.split() if s.startswith("http")})
            if source_links:
                st.subheader("🔗 Sources:")
                for link in source_links[:3]:
                    st.markdown(f"🔹 [{link}]({link})")

# ==========================================================
# 🦶 FOOTER
# ==========================================================
st.markdown(
    """
    <hr>
    <div style="text-align:center; color:gray;">
    Built with using <b>LangChain</b> and <b>Streamlit</b> | Designed by <b>Opeyemi Ojajuni</b>

    <hr>
    <p style='text-align: center; color: gray;'>
        🔗 <a href="https://github.com/jajupeter/insightBot" target="_blank">View Source Code on GitHub</a>
    </p>
    </div>
    """,
    unsafe_allow_html=True,
)