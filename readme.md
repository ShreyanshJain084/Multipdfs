# Multi-PDF Chatbot using LLMs

## 📌 Project Overview
This project is a **Multi-PDF Chatbot** that allows users to upload multiple PDFs and interact with them using an **LLM (Mistral-7B-Instruct)** without requiring an API key. The chatbot utilizes **LangChain, FAISS vector search, and Sentence Transformers** to generate meaningful responses from uploaded documents.

## 🚀 Features
- 📂 **Upload Multiple PDFs**: Supports multiple document inputs.
- 🧠 **Offline LLM**: Uses **Mistral-7B-Instruct-GGUF** (No API key required).
- 🔍 **FAISS Vector Search**: Efficiently searches document content.
- 📝 **Conversational Memory**: Keeps track of chat history.
- 🎛️ **Streamlit UI**: Simple and interactive web-based interface.

## 🛠️ Tech Stack
- **Python** 🐍
- **Streamlit** (UI Framework)
- **LangChain** (Conversational AI Framework)
- **FAISS** (Vector Search)
- **CTransformers** (Offline LLM Model Loader)
- **Sentence-Transformers** (Embedding Model)

## 🏗️ Installation
1. **Clone the Repository**
   ```bash
   git clone https://github.com/RajShreyanshu28/Multipdf.git
   cd Multi-PDF-Chatbot
   ```
2. **Create a Virtual Environment (Optional but Recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Download & Set Up Mistral-7B-GGUF Model**
   ```bash
   mkdir models
   cd models
   wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf
   ```

## ▶️ Running the Application
```bash
streamlit run app.py
```
Then open your browser and navigate to **http://localhost:8501**.

## 🏆 Results
- Successfully processes multiple PDFs.
- Uses **Mistral-7B for answering queries** efficiently.
- Achieves **high response quality without API keys**.

## 🤝 Contributions
Feel free to fork the repository and submit a pull request.

## 📜 Author
Shreyansh Jain

