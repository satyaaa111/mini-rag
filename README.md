# Mini RAG Assistant

A Retrieval-Augmented Generation (RAG) system built for any environment. This AI assistant answers user questions using internal documents (policies, FAQs, specifications) rather than relying on the model's general knowledge.

---

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Grounding & Transparency](#grounding--transparency)
- [Findings and Observations] (#findings--observations)


---

## Features

- **Document Upload**: Upload PDF/TXT documents dynamically via the frontend
- **Semantic Search**: Hybrid retrieval (Vector + Keyword) for accurate results
- **Multi-Query Retrieval**: Generates 4 query variations to improve recall
- **Reciprocal Rank Fusion (RRF)**: Merges results from multiple search methods
- **Reranking**: Uses BGE Reranker to select top-k most relevant chunks
- **Local LLM**: Powered by Ollama (phi3) - no API costs
- **Transparency**: Displays retrieved document chunks for every answer
- **Chat History**: Context-aware conversations

---

## Tech Stack

| Component | Technology | Why |
| :--- | :--- | :--- |
| **Embedding Model** | `BAAI/bge-small-en-v1.5` | High performance, lightweight (384 dimensions), local |
| **Vector Database** | `FAISS` (CPU) | Fast semantic search, local storage |
| **Keyword Search** | `BM25` (rank-bm25) | Hybrid search for better recall |
| **Reranker** | `BAAI/bge-reranker-small` | Improves relevance of final chunks |
| **LLM** | `phi3` (via Ollama) | Local, fast, follows grounding instructions |
| **Backend** | `FastAPI` (Python) | Async API, easy integration with ML libraries |
| **Frontend** | `Next.js` (JavaScript) | Modern UI, responsive, custom components |
| **Text Splitter** | `RecursiveCharacterTextSplitter` | Preserves document structure during chunking |

---

## Project Structure
 Can be seen in the github itself


## Setup Instructions

### Prerequisites

- **Python 3.10 or 3.11** (Required - Python 3.12+ has compatibility issues)
- **Node.js 18+**
- **Ollama** installed ([Download](https://ollama.com))



## 1. Ollama Setup

### Pull the required model
ollama pull phi3
### Ensure Ollama is running
ollama serve

---

## 2. Backend Setup  

### Navigate to the backend directory
cd backend
### Create and activate a virtual environment
python -m venv venv
### Windows:
.\venv\Scripts\activate
### macOS/Linux:
source venv/bin/activate
### Install required dependencies
pip install -r requirements.txt
### Start the FastAPI server
python -m app.main  

---

## 3. Frontend Setup  

### Navigate to the frontend directory
cd frontend
### Install dependencies
npm install
### Start the development server
npm run dev  

---

## Grounding & Transparency  

### How Grounding is Enforced

The LLM is explicitly instructed via system prompt to use only retrieved context:

- **system_prompt**

ROLE: You are a strict document assistant. Follow the RULES in every answer you generate.

RULES:
1. Use ONLY the provided Context to answer.
2. If the user greets you, respond politely but do not retrieve documents.
3. If the Context does not contain the answer, or the question is unrelated to the context, 
   state: "I am designed to assist you on relevant questions. Please ask within the context of the documents provided."
4. Do NOT use your internal knowledge to answer general questions (e.g., about animals, food, music, films, sexual things, philosophy, general politics, physics, chemistry, mathematics etc).
5. Answer should not be long unnecessarily.
6. Avoid unsupported claims. If the provided data does not give enough information, do not give gibberish or forced answers.
### Transparency Display
Every answer shows the retrieved document chunks used:
1. Click "View Retrieved Context" below any AI response
2. See exact text chunks + source filenames
3. Verify the answer is grounded in the displayed content
This satisfies the assignment's Transparency and Explainability requirement.

---

## Findings and Ovservation

- **Latency** : Local phi3 typically has lower latency (1s–3s) as there is no network overhead, whereas OpenRouter depends on internet speed (2s–5s).
- **Groundedness** : Smaller local models (3B) are more prone to "forgetting" the negative constraint ("If not found, say I don't know") compared to larger cloud models.
- **Quality** : Cloud models handle complex table-based specifications (like the Package Comparison in doc2.pdf) with better formatting.

