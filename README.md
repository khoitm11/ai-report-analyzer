# 🤖 AI Report Analyzer

An intelligent application that uses a Retrieval-Augmented Generation (RAG) pipeline to analyze and summarize PDF documents like annual reports, financial statements, or research papers.

<!-- You can add a screenshot or GIF of your app in action here! It makes the README look much better. -->
<!-- ![App Demo](link_to_your_gif_or_screenshot.png) -->

## ✨ Features

-   **Analyze Any PDF:** Upload long and dense documents for quick analysis.
-   **Automated Summaries:** Get the key takeaways from a document in seconds.
-   **In-Depth Analysis:** Perform a SWOT (Strengths, Weaknesses, Opportunities, Threats) or Risk Analysis based on the document's content.
-   **Fast & Efficient:** Caches processed documents, so subsequent analyses on the same file are instantaneous.

## 🛠️ Tech Stack

-   **AI Model:** Google Gemini (`gemini-1.5-flash-latest`)
-   **Framework:** LangChain for orchestrating the RAG pipeline.
-   **User Interface:** Gradio for a simple and interactive web UI.
-   **Vector Database:** FAISS (`faiss-cpu`) for efficient local similarity search.
-   **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
-   **PDF Processing:** `pypdf`

## 🚀 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

-   Python 3.9+
-   A Google API Key for using the Gemini model. You can get one from [Google AI Studio](https://aistudio.google.com/app/apikey).

### Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```sh
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

4.  **Set up your environment variables:**
    Create a file named `.env` in the root of the project and add your Google API key:
    ```
    GOOGLE_API_KEY="YOUR_API_KEY_HERE"
    ```

5.  **Run the application:**
    ```sh
    python app.py
    ```

    Open your web browser and navigate to the local URL provided in the terminal (usually `http://127.0.0.1:7860`).

## 🏛️ How It Works: The RAG Pipeline

This application leverages a powerful RAG (Retrieval-Augmented Generation) pipeline to ensure the AI's responses are grounded in the provided document.

1.  **Load & Chunk:** The uploaded PDF is loaded and split into smaller, manageable text chunks.
2.  **Embed & Store:** Each chunk is converted into a numerical vector (embedding) and stored in a high-speed FAISS vector database. This database is cached locally to make future use of the same document instantaneous.
3.  **Retrieve & Generate:** When you request an analysis, the system retrieves the most relevant chunks from the database and provides them to the Gemini model as context. This allows it to generate an accurate, document-aware response.
