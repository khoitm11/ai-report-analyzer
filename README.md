---
title: AI Report Analyzer
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
---

# 🤖 AI Report Analyzer

This project is an intelligent app designed to tackle the pain of reading dense PDF documents. Feed it an annual report, a research paper, or any long PDF, and it leverages a Retrieval-Augmented Generation (RAG) pipeline to give you back the insights you need, fast.

## 🚀 How It Works

1.  **Upload a PDF** using the "Control Panel" on the left.
2.  **Wait for the processing to finish.** The "Status" will change to **Ready!** when it's done.
    *   *Heads up:* The first time you upload a document, it might take a moment to build the knowledge base. Subsequent analyses of the same file will be nearly instant thanks to caching.
3.  **Click an analysis button** (e.g., 📄 Summary, 📊 SWOT Analysis, ⚠️ Risk Analysis).
4.  **Check out the results** in the "Results" section at the bottom.

## 🛠️ Tech Stack

-   **Core AI:** Google Gemini (`gemini-1.5-flash-latest`) - The brain of the operation.
-   **Orchestration Framework:** LangChain - For chaining everything together.
-   **User Interface:** Gradio - To spin up a simple and clean web UI.
-   **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` - For turning text into meaningful vectors.
-   **Vector Database:** FAISS (`faiss-cpu`) - A super-fast, local vector store. No cloud dependencies!
-   **PDF Processing:** `pypdf` - For wrestling with text extraction from PDFs.

## 🏛️ The RAG Architecture Explained

This application is powered by a RAG pipeline, which is a fancy way of saying it's smart about how it reads your document. Here’s the breakdown:

1.  **Load & Chunk:** First, the app ingests the uploaded PDF and breaks it down into smaller, more manageable chunks of text. This helps the AI focus on specific details.
2.  **Embed & Store:** Each text chunk is converted into a numerical vector (an "embedding") and stored in a high-speed FAISS vector database. This database is then cached on the server, making future requests for the same document incredibly fast.
3.  **Retrieve & Generate:** When you request an analysis, the system searches the vector database to find the most relevant chunks from the document. These chunks are then provided to the Gemini model as context, allowing it to generate an accurate, document-aware response instead of just making things up.
