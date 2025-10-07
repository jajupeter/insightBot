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
#from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS  # ✅ Use langchain_community for FAISS

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

# ==========================================================
# 🔑 API KEY INPUT
# ==========================================================

#st.sidebar.title("⚙️ Settings")

#with st.sidebar.expander("🔑 OpenAI API Key (Optional)", expanded=False):
    #user_api_key = st.text_input("Enter your OpenAI API Key:", type="password")
    #if user_api_key:
        #os.environ["OPENAI_API_KEY"] = user_api_key
       # st.success("✅ API key loaded successfully!")
    #else:
        #st.warning("⚠️ Enter your OpenAI API key to enable full functionality.")

# Load environment variables if any
load_dotenv()

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

       Whether you’re conducting academic research, analyzing market trends, or just staying ahead of the news cycle — InsightBot turns data into knowledge and knowledge into action ⚡ 
    </ul>
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

