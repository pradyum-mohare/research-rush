# 📄 Research Rush

A multi-PDF query system that lets you upload research documents and ask natural language questions — receiving precise, context-aware answers powered by **Gemini LLM**, **Pinecone vector database**, and **RAG (Retrieval-Augmented Generation)** with **conversational memory**.

---

## 🚀 Demo

Upload any PDF → Ask questions → Get grounded, accurate answers with follow-up support.

---

## 🧠 How It Works

```
PDFs → Text Extraction + OCR → Chunking → Gemini Embeddings → Pinecone (Vector DB)
                                                                        ↓
User Query → Embed Query → Pinecone Similarity Search → Top-K Chunks
                                                                        ↓
                                                    Gemini LLM + Context → Answer
                                                    (+ Conversational Memory)
```

1. **PDF Ingestion** — Extracts native text using PyPDF2. Pages containing embedded images or diagrams are automatically detected via PyMuPDF and processed through Tesseract OCR, ensuring no content is missed even in diagram-heavy documents.

2. **Chunking** — Extracted text is split into 1,000-character overlapping chunks (200-char overlap) using LangChain's `RecursiveCharacterTextSplitter`.

3. **Embedding** — Each chunk is converted into a 3,072-dimensional semantic vector using Google's `gemini-embedding-001` model.

4. **Vector Storage** — Embeddings are stored in Pinecone's serverless vector database for fast similarity search.

5. **RAG Generation** — At query time, the question is embedded and Pinecone retrieves the top-5 most relevant chunks. These are injected as grounded context into a prompt sent to Gemini 2.5 Flash, which generates a precise answer restricted to the uploaded documents — preventing hallucination.

6. **Conversational Memory** — Full chat history is maintained across turns and injected into every prompt, so follow-up questions like *"what are its disadvantages?"* resolve correctly without re-stating the topic.

---

## 🛠️ Tech Stack

| Technology | Role |
|---|---|
| Gemini 2.5 Flash | LLM for answer generation |
| Gemini Embedding (gemini-embedding-001) | Text → vector conversion |
| Pinecone (Serverless) | Vector storage + similarity search |
| LangChain | Text splitting + embedding wrapper |
| PyMuPDF (fitz) | PDF page rendering + image detection |
| PyPDF2 | Native PDF text extraction |
| Tesseract OCR | Text extraction from image-based pages |
| Streamlit | Web UI |
| Python-dotenv | Environment variable management |

---

## 📁 Project Structure

```
research-rush/
├── app.py               # Streamlit UI — upload, process, chat
├── pdf_processor.py     # PDF extraction + per-page OCR + chunking
├── vector_store.py      # Pinecone index setup, embedding, upsert, query
├── rag_chain.py         # Gemini RAG pipeline + conversational memory
├── requirements.txt
└── .gitignore
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/pradyum-mohare/research-rush.git
cd research-rush
```

### 2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Install system dependencies

**Tesseract OCR (Windows):**
Download and install from https://github.com/UB-Mannheim/tesseract/wiki

Update the path in `pdf_processor.py`:
```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

### 5. Set up API keys

Create a `.env` file in the root directory:
```
GOOGLE_API_KEY=your_gemini_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

- Get your Gemini API key: https://aistudio.google.com/app/apikey
- Get your Pinecone API key: https://app.pinecone.io

### 6. Run the app
```bash
streamlit run app.py
```

---

## 💬 Usage

1. Open the app in your browser at `http://localhost:8501`
2. Upload one or more PDF files using the sidebar
3. Click **⚡ Process PDFs** — text is extracted, embedded, and indexed
4. Ask any question in the chat box
5. Use follow-up questions freely — memory is active across the conversation
6. Click **🗑️ Clear conversation** in the sidebar to start fresh

---

## 🔑 Key Design Decisions

**Hybrid OCR strategy** — Rather than OCR-ing the whole document or relying solely on a text-length threshold, the system uses PyMuPDF to detect embedded image objects per page. Any page containing images gets both native text extraction AND OCR, ensuring text inside diagrams isn't silently lost even when the surrounding page has plenty of regular text.

**RAG over direct upload** — Uploading full documents to Gemini on every query hits context window limits, costs significantly more per query, and degrades answer quality on large document sets due to the "lost in the middle" problem. RAG retrieves only the most relevant chunks (~5) per query, keeping prompts lean and answers precise regardless of document size.

**Grounded prompting** — The system prompt explicitly instructs Gemini to answer only from the provided context and admit when information is unavailable, preventing hallucination and keeping answers tied to the actual uploaded documents.

---

## 📋 Requirements

- Python 3.9+
- Tesseract OCR installed locally
- Google Gemini API key (free tier available)
- Pinecone API key (free tier available)

---

## 👤 Author

**Pradyum Mohare** 
[GitHub](https://github.com/pradyum-mohare)
