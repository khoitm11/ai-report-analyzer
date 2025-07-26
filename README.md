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

An intelligent application that uses a Retrieval-Augmented Generation (RAG) pipeline to analyze PDF documents like annual reports, financial statements, or research papers.

## 🚀 How to Use

1.  **Upload a PDF file** in the "Control Panel" on the left.
2.  **Wait for the processing to complete.** The "Status" will update to "**Ready!**".
    *   *Note: The first time you upload a file, this may take a moment while the knowledge base is being built. Subsequent uploads of the same file will be almost instant thanks to caching.*
3.  **Click an analysis button** (📄 Summary, 📊 SWOT Analysis, ⚠️ Risk Analysis).
4.  **View the result** in the "Results" section at the bottom.

## 🛠️ Technology Stack

-   **Core AI:** Google Gemini (`gemini-1.5-flash-latest`)
-   **Orchestration Framework:** LangChain
-   **User Interface:** Gradio
-   **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`
-   **Vector Database:** FAISS (`faiss-cpu`)
-   **PDF Processing:** `pypdf`

## 🏛️ About the RAG Architecture

This application leverages a powerful RAG pipeline:

1.  **Load & Chunk:** The uploaded PDF is loaded and split into smaller, manageable text chunks.
2.  **Embed & Store:** Each chunk is converted into a numerical vector (embedding) and stored in a high-speed FAISS vector database. This database is then cached locally on the server to make future use of the same document instantaneous.
3.  **Retrieve & Generate:** When you request an analysis, the system retrieves the most relevant chunks from the database and provides them to the Gemini model as context, allowing it to generate an accurate, document-aware response.