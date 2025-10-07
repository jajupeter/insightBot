# -*- coding: utf-8 -*-
"""
Created on 06 Oct 2025
Author: Jaju Peter
"""

import os
import time
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAI, OpenAIEmbeddings
import nltk

# ==========================================================
# 📚 DOWNLOAD NLTK RESOURCES
# ==========================================================
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

# ==========================================================
# 🎨 PAGE CONFIGURATION
# ==========================================================
st.set_page_config(
    page_title="InsightBot: Research Tool",
    page_icon="📈",
    layout="wide",
)

# ==========================================================
# 💅 CUSTOM STYLING
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
# 🔑 LOAD API KEY (SECURELY)
# ==========================================================
load_dotenv()

if "OPENAI_API_KEY" in st.secrets:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
else:
    openai_api_key = st.sidebar.text_input("🔑 Enter your OpenAI API Key", type="password")

if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
else:
    st.sidebar.warning("⚠️ Please enter your OpenAI API key or set it in Streamlit Secrets.")
    st.stop()

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
        <li>Map connections and uncover deeper meaning using advanced AI embeddings ⚙️</li>
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
# 🧠 MAIN LOGIC WITH PROGRESS DISPLAY
# ==========================================================
main_placeholder = st.empty()

if process_url_clicked:
    if not urls:
        st.warning("⚠️ Please enter at least one valid URL.")
    else:
        with st.spinner("🔍 Fetching and processing data... Please wait."):
            try:
                # 🟢 1️⃣ Load Data
                main_placeholder.text("📥 Data Loading... Started... ⏳")
                loader = WebBaseLoader(urls)
                data = loader.load()
                main_placeholder.text("📥 Data Loading... Completed ✅✅✅")

                if not data:
                    st.error("❌ No data found. Ensure URLs are publicly accessible and contain text.")
                    st.stop()

                # 🟢 2️⃣ Split Data
                main_placeholder.text("✂️ Text Splitting... Started... ⏳")
                text_splitter = RecursiveCharacterTextSplitter(
                    separators=["\n\n", "\n", ".", ","],
                    chunk_size=1000
                )
                docs = text_splitter.split_documents(data)
                main_placeholder.text(f"✂️ Text Splitting... Completed ✅✅✅ — {len(docs)} chunks created")

                if not docs or len(docs) == 0:
                    st.error("❌ No text chunks created. Check if URLs contain readable text.")
                    st.stop()

                # 🟢 3️⃣ Create Embeddings
                main_placeholder.text("🧠 Building Embedding Vectors... Started... ⏳")
                embeddings = OpenAIEmbeddings()
                vectorstore_openai = FAISS.from_documents(docs, embeddings)
                main_placeholder.text("🧠 Building Embedding Vectors... Completed ✅✅✅")

                # 🟢 4️⃣ Save FAISS Index
                main_placeholder.text("💾 Saving FAISS Index... ⏳")
                vectorstore_openai.save_local(index_path)
                main_placeholder.text("💾 FAISS Index Saved Successfully ✅✅✅")

                st.success("✅ Articles processed and stored successfully! You can now ask InsightBot questions below 👇")

            except Exception as e:
                st.error(f"⚠️ An error occurred during processing: {e}")

# ==========================================================
# 💬 QUERY SECTION WITH PROGRESS DISPLAY
# ==========================================================
query = st.text_input("🔎 Ask a question about the articles:")

if query:
    if not os.path.exists(index_path):
        st.error("❌ No FAISS index found. Please process URLs first.")
    else:
        try:
            query_placeholder = st.empty()

            # 🟢 1️⃣ Load FAISS Index
            query_placeholder.text("📚 Loading FAISS Index... ⏳")
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.load_local(
                index_path,
                embeddings,
                allow_dangerous_deserialization=True
            )
            query_placeholder.text("📚 FAISS Index Loaded ✅✅✅")

            # 🟢 2️⃣ Initialize LLM
            query_placeholder.text("🤖 Initializing AI Model... ⏳")
            llm = OpenAI(temperature=0.7, max_tokens=500)
            query_placeholder.text("🤖 AI Model Ready ✅✅✅")

            # 🟢 3️⃣ Retrieve and Generate Answer
            query_placeholder.text("🔍 Retrieving relevant documents... ⏳")
            chain = RetrievalQAWithSourcesChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever()
            )
            result = chain({"question": query}, return_only_outputs=True)
            query_placeholder.text("🧩 Generating Answer... ✅✅✅")

            # 🟢 4️⃣ Display Answer
            st.success("✅ Done!")
            st.header("📘 Answer")
            st.write(result["answer"])

            sources = result.get("sources", "")
            if sources:
                st.subheader("🔗 Sources:")
                for link in list({s.strip().strip(',') for s in sources.split() if s.startswith("http")})[:3]:
                    st.markdown(f"🔹 [{link}]({link})")

        except Exception as e:
            st.error(f"⚠️ Error while answering query: {e}")

# ==========================================================
# 🦶 FOOTER
# ==========================================================
st.markdown(
    """
    <hr>
    <div style="text-align:center; color:gray;">
    Built with ❤️ using <b>LangChain</b> and <b>Streamlit</b> | Designed by <b>Opeyemi Ojajuni</b>
    <hr>
    <p style='text-align: center; color: gray;'>
        🔗 <a href="https://github.com/jajupeter/insightBot" target="_blank">View Source Code on GitHub</a>
    </p>
    </div>
    """,
    unsafe_allow_html=True,
)