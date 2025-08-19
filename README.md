# AI FAQ Assistant

An AI-powered FAQ assistant that uses embeddings and semantic search to answer frequently asked questions.  
The system ingests FAQs into a vector database and retrieves the most relevant response based on user queries.

---

## 🚀 Features
- Store and manage FAQs
- Search answers using semantic similarity
- Fast and accurate retrieval
- Easy to extend with more data

---

## 🛠️ Installation & Setup

1. **Clone the repository**
    git clone https://github.com/Syed-Burhan0684/Customer_Support_Agent.git
    cd Customer_Support_Agen

----
2.**Create a virtual environment**
    python -m venv .venv
    source .venv/bin/activate   # (Linux/Mac)
    .venv\Scripts\activate      # (Windows)

----
3.**Install dependencies**

    pip install -r requirements.txt
----
4.**Ingest FAQ data**

    python src/ingest_faq.py

----
5.**Run the FastAPI server**

    uvicorn src.app:app --reload --port 8000
----
6.**Usage**

Open your browser at http://localhost:8000/static/index.html

📌 **Example**

Question:

How can I reset my password?


Answer:

To reset your password go to Settings → Reset Password...

