import gradio as gr
import os
import hashlib
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI

#Config
CACHE_DIR = Path("FAISS_INDEX_CACHE")
CACHE_DIR.mkdir(exist_ok=True)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Dòng này sẽ đọc key từ Hugging Face secrets khi triển khai
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

#Language Model
llm = None
if not GOOGLE_API_KEY:
    print("API Key not found. Please set GOOGLE_API_KEY in secrets.")
else:
    try:
        llm = GoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=GOOGLE_API_KEY)
    except Exception as e:
        print(f"LLM initialization error: {e}")

#Core Logic
def get_pdf_hash(pdf_path):
    with open(pdf_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def process_pdf_and_build_retriever(pdf_file_path, progress=gr.Progress(track_tqdm=True)):
    if not pdf_file_path:
        return None, "Status: No PDF file provided."

    pdf_hash = get_pdf_hash(pdf_file_path)
    index_path = CACHE_DIR / f"{pdf_hash}.faiss"

    progress(0, desc="Checking for cached Knowledge Base...")
    if index_path.exists():
        try:
            embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
            vector_store = FAISS.load_local(str(index_path), embeddings, allow_dangerous_deserialization=True)
            progress(1, desc="Knowledge Base loaded from cache.")
            return vector_store.as_retriever(search_kwargs={"k": 5}), "Status: **Ready!** You can now perform an analysis."
        except Exception as e:
            return None, f"Status: Error loading cached index: {e}"

    progress(0.2, desc="Creating new Knowledge Base...")
    try:
        loader = PyPDFLoader(pdf_file_path)
        documents = loader.load()

        progress(0.4, desc="Splitting document...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)

        progress(0.6, desc="Embedding text (this might take a moment)...")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

        vector_store = FAISS.from_documents(texts, embeddings)
        vector_store.save_local(str(index_path))

        progress(1, desc="Knowledge Base created and cached.")
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        return retriever, "Status: **Ready!** You can now perform an analysis."
    except Exception as e:
        return None, f"Status: Error creating Knowledge Base: {e}"

def generate_analysis(retriever, analysis_type):
    if retriever is None:
        gr.Warning("Knowledge Base is not ready. Please upload and process a PDF file first.")
        return "Please upload and process a PDF file before running an analysis."
    if not llm:
        gr.Warning("Language Model (LLM) is not initialized. Please check your API Key.")
        return "LLM not available. Check API Key configuration."

    prompt_templates = {
        "Summary": "As an expert business analyst, provide a concise, professional summary of the key points from the provided document.\nContext: {context}\nQuestion: What are the main takeaways from this document?\nAnswer:",
        "SWOT Analysis": "You are a strategic business consultant. Based on the information provided, conduct a detailed SWOT analysis (Strengths, Weaknesses, Opportunities, Threats).\nContext: {context}\nQuestion: What is the SWOT analysis based on this document?\nAnswer:",
        "Risk Analysis": "As a risk management expert, identify and list the main financial, operational, and market risks mentioned in the document. Explain each briefly.\nContext: {context}\nQuestion: What are the key risks identified in this document?\nAnswer:"
    }

    prompt = PromptTemplate(template=prompt_templates[analysis_type], input_variables=["context", "question"])
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs={"prompt": prompt})
    question = {
        "Summary": "Summarize the key points of this document.",
        "SWOT Analysis": "Perform a SWOT analysis based on this document.",
        "Risk Analysis": "Identify the key risks mentioned in this document."
    }[analysis_type]

    try:
        result = qa_chain({"query": question})
        return result["result"]
    except Exception as e:
        return f"An error occurred during analysis: {e}"

#Gradio
with gr.Blocks(theme=gr.themes.Default(primary_hue="blue"), title="AI Report Analyzer") as demo:
    retriever_state = gr.State(None)

    gr.Markdown("# 🤖 AI Report Analyzer")
    gr.Markdown("Upload a PDF to build a Knowledge Base, then click a button to perform analysis.")

    with gr.Row(equal_height=True):
        with gr.Column(scale=1, min_width=350):
            with gr.Group():
                gr.Markdown("### 1. Control Panel")
                pdf_upload = gr.File(label="Upload PDF Report", file_types=[".pdf"])
                status_output = gr.Markdown("Status: Waiting for PDF file...")

        with gr.Column(scale=2):
            with gr.Group():
                gr.Markdown("### 2. Analysis Actions")
                with gr.Row():
                    summary_btn = gr.Button("📄 Summary", interactive=False)
                    swot_btn = gr.Button("📊 SWOT Analysis", interactive=False)
                    risk_btn = gr.Button("⚠️ Risk Analysis", interactive=False)

    with gr.Row():
        with gr.Column():
            with gr.Group():
                gr.Markdown("### 3. Results")
                output_textbox = gr.Markdown("Analysis results will appear here...")

    def handle_pdf_upload(pdf_file):
        if pdf_file:
            retriever, status_msg = process_pdf_and_build_retriever(pdf_file.name)
            if retriever:
                return retriever, status_msg, gr.Button(interactive=True), gr.Button(interactive=True), gr.Button(interactive=True)
        btn_update = gr.Button(interactive=False)
        return None, "Status: Waiting for PDF file...", btn_update, btn_update, btn_update

    pdf_upload.upload(
        handle_pdf_upload,
        inputs=[pdf_upload],
        outputs=[retriever_state, status_output, summary_btn, swot_btn, risk_btn]
    )

    summary_btn.click(lambda r: generate_analysis(r, "Summary"), [retriever_state], [output_textbox])
    swot_btn.click(lambda r: generate_analysis(r, "SWOT Analysis"), [retriever_state], [output_textbox])
    risk_btn.click(lambda r: generate_analysis(r, "Risk Analysis"), [retriever_state], [output_textbox])

if __name__ == "__main__":
    demo.launch()