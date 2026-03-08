# Hallucination Detection System

A complete system for detecting hallucinations in Large Language Model (LLM) responses using Retrieval-Augmented Generation (RAG).

## 🎯 Project Overview

This system detects when an LLM generates information that is not supported by or contradicts the knowledge base. It uses:

1. **RAG (Retrieval-Augmented Generation)** - Retrieves relevant documents to ground LLM responses
2. **Claim Extraction** - Extracts factual claims from LLM responses
3. **Fact Verification** - Verifies each claim against retrieved evidence
4. **Hallucination Scoring** - Provides a score and classification for response reliability

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Query                                │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              RAG Pipeline                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  Embeddings │─▶│  ChromaDB   │─▶│  Relevant Context   │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              LLM Generation (Google Gemini)                  │
│              Generates response with context                 │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              Hallucination Detection                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Extract   │─▶│   Verify    │─▶│   Score & Report    │  │
│  │   Claims    │  │   Claims    │  │                     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
HALLUCINATION DETECTION/
├── app.py                      # Streamlit web application
├── config.py                   # Configuration settings
├── setup.py                    # Setup and initialization script
├── requirements.txt            # Project dependencies
├── README.md                   # This file
│
├── data/
│   └── knowledge_base.txt      # Knowledge base documents
│
├── knowledge_base/
│   ├── __init__.py
│   ├── document_loader.py      # Document loading and chunking
│   ├── embeddings.py           # Embedding model wrapper
│   └── vector_store.py         # ChromaDB vector storage
│
├── rag/
│   ├── __init__.py
│   ├── retriever.py            # Document retrieval
│   └── generator.py            # LLM-based generation
│
└── detection/
    ├── __init__.py
    ├── claim_extractor.py      # Extract claims from responses
    ├── fact_verifier.py        # Verify claims against evidence
    └── hallucination_detector.py # Main detection pipeline
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd "e:\HALLUCINATION DETECTION"
pip install -r requirements.txt
```

### 2. Run Setup (First Time Only)

```bash
python setup.py
```

This will:
- Load the embedding model
- Build the vector database
- Test all components

### 3. Run the Web Application

```bash
streamlit run app.py
```

The app will open at http://localhost:8501

## 🔧 Configuration

Edit `config.py` to customize:

- `GOOGLE_API_KEY` - Your Google Gemini API key
- `LLM_MODEL` - Gemini model to use
- `EMBEDDING_MODEL` - Sentence transformer model
- `CHUNK_SIZE` - Document chunk size
- `TOP_K_DOCUMENTS` - Number of documents to retrieve

## 📊 How It Works

### 1. Document Processing
- Documents are loaded from `data/knowledge_base.txt`
- Text is split into chunks using RecursiveCharacterTextSplitter
- Chunks are embedded using sentence-transformers
- Embeddings are stored in ChromaDB

### 2. Query Processing
- User submits a query
- Relevant documents are retrieved using semantic search
- Retrieved context is passed to the LLM along with the query

### 3. Response Generation
- LLM generates a response grounded in the retrieved context
- Prompt instructs the LLM to only use information from context

### 4. Hallucination Detection
- Claims are extracted from the response
- Each claim is verified against the context
- Claims are classified as: SUPPORTED, CONTRADICTED, or NOT_ENOUGH_INFO
- Overall hallucination score is calculated

### 5. Results Display
- Response is shown with hallucination score
- Each claim is highlighted with its verification status
- Risk level is displayed (LOW, MEDIUM, HIGH)

## 🎨 Web Interface Features

- **Beautiful gradient UI** with modern styling
- **Real-time analysis** of LLM responses
- **Interactive claim verification** display
- **Visual score cards** with color-coded risk levels
- **Expandable context view** showing retrieved documents
- **Example queries** for quick testing

## 📈 Understanding Results

### Hallucination Score (0-100%)
- **0-30%**: LOW risk - Response is well-grounded in evidence
- **30-70%**: MEDIUM risk - Some claims may lack support
- **70-100%**: HIGH risk - Response contains unsupported claims

### Claim Verdicts
- ✅ **SUPPORTED**: Claim is verified by the knowledge base
- ❌ **CONTRADICTED**: Claim conflicts with the knowledge base
- ❓ **NOT_ENOUGH_INFO**: Knowledge base doesn't cover this claim

## 📚 Adding Your Own Knowledge Base

1. Add text files to the `data/` directory
2. Run the setup script again to rebuild the vector store
3. Or click "Rebuild Knowledge Base" in the web app sidebar

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| LLM | Google Gemini API |
| Embeddings | sentence-transformers |
| Vector Database | ChromaDB |
| Web Framework | Streamlit |
| Text Processing | LangChain |

## 📝 License

This project is for educational purposes (Final Year Project).

## 🙏 Acknowledgments

- Google Gemini for the LLM API
- Hugging Face for sentence-transformers
- ChromaDB for vector storage
- Streamlit for the web framework
