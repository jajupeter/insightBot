# 📈 InsightBot: News Research Tool  
**Streamlit • Python • LangChain • OpenAI API**  

---

## 📌 Project Overview  
**InsightBot** is an AI-powered research assistant that automates the process of analyzing and summarizing online news or research articles. Built with **Streamlit** and **LangChain**, it extracts insights, identifies relationships between topics, and answers user questions with source-backed responses — turning raw text into actionable knowledge.  

This app demonstrates how modern **Large Language Model (LLM)** pipelines can be integrated into an intuitive, web-based research tool that bridges human curiosity and machine reasoning.  

🔗 **Live Demo:** [Streamlit App (InsightBot)](https://insightbotnewsresearch.streamlit.app/)  

---

## ❓ Problem Statement  
In today’s fast-paced digital landscape, professionals, researchers, and students are overwhelmed by the sheer volume of online content. Traditional reading and note-taking methods are:  

- ⏳ Time-consuming  
- ⚖️ Prone to cognitive overload and bias  
- 📉 Inefficient for comparing multiple sources  

**InsightBot** addresses this by automatically:  
- Fetching content from online articles  
- Segmenting and embedding the text into AI-readable vectors  
- Using OpenAI’s LLM to retrieve, reason, and summarize findings with citation links  

---

## 🔬 Methodology – Approach  

### Data Collection  
- **Input**: Up to three article URLs provided by the user  
- **Loader**: `UnstructuredURLLoader` (LangChain) fetches and cleans the content  

### Text Processing  
- **Splitter**: `RecursiveCharacterTextSplitter` divides the article text into manageable semantic chunks for embedding  
- **Embedding**: `OpenAIEmbeddings` encodes chunks into numerical vector representations  

### Vector Storage  
- **Database**: `FAISS` (Facebook AI Similarity Search) stores embeddings locally for fast retrieval  
- **Retrieval**: Similarity search fetches the most relevant text chunks for user queries  

### Question Answering  
- **LLM Engine**: `OpenAI` via LangChain’s `RetrievalQAWithSourcesChain`  
- **Output**: Concise answers with supporting source URLs  

### Deployment  
- Hosted on **Streamlit Cloud**  
- Interactive sidebar for URL input and OpenAI API key configuration  

---

## ⚙️ Tech Stack  

| Category | Tools / Libraries |
|-----------|------------------|
| **Programming Language** | Python 3.11 |
| **Framework** | Streamlit |
| **LLM Integration** | LangChain, OpenAI API |
| **Embeddings & Retrieval** | FAISS |
| **Web Parsing** | Unstructured, BeautifulSoup |
| **Environment Management** | python-dotenv |
| **Deployment** | Streamlit Cloud |
| **Version Control** | Git + GitHub |

---

## ✨ App Features  

- 🧠 **AI-Powered Insight Extraction** – Summarizes and connects information across multiple sources  
- 🔍 **Smart Q&A Interface** – Ask research-style questions and get answers with references  
- 📎 **Multi-URL Input** – Compare perspectives from up to three online articles  
- 💾 **Local Vector Index** – Efficient document search using FAISS embeddings  
- 🎨 **Dark Mode UI** – Modern design built with custom CSS and Streamlit themes  
- 🔐 **Secure API Key Handling** – Optional user-provided OpenAI API key for full functionality  
- 📘 **Source Transparency** – Displays clickable article links for verification  

---

## 📜 License  
This project is released under the **MIT License**.  

You are free to use, modify, and distribute the code for educational or research purposes with proper attribution.  

---

## 👨‍💻 Author  
**Dr. Opeyemi “Jaju” Ojajuni**  
Research Scientist | AI & XR Learning Innovation | Southern University  
🔗 [GitHub Repository](https://github.com/jajupeter/insightBot)  
