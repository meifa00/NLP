import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
import csv
import os
import xlrd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
import time
import requests

def fetch_files_from_urls(url_input):
    """
    Fetches files from the provided URLs.
    """
    urls = [url.strip() for url in url_input.split(",")]
    fetched_files = []
    url_errors = []
    for url in urls:
        try:
            response = requests.get(url)
            response.raise_for_status()
            file_name = url.split("/")[-1]
            with open(file_name, "wb") as f:
                f.write(response.content)
            fetched_files.append(file_name)
        except requests.RequestException as e:
            url_errors.append((url, str(e)))
    return fetched_files, url_errors

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")


# ===========================
# Helper Functions
# ===========================

def get_text_from_file(file, file_type):
    """
    Extracts text from the given file based on its type.
    """
    if file_type == "pdf":
        return get_pdf_text([file])
    elif file_type == "docx":
        return get_docx_text([file])
    elif file_type == "txt":
        return get_txt_text([file])
    elif file_type == "csv":
        return get_csv_text([file])
    elif file_type == "xls":
        return get_xls_text([file])
    elif file_type == "html":
        return get_html_text([file])
    return ""


def get_pdf_text(pdf_docs):
    """
    Extracts text from PDF files.
    """
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_docx_text(docx_docs):
    """
    Extracts text from DOCX files.
    """
    text = ""
    for docx in docx_docs:
        doc = Document(docx)
        for para in doc.paragraphs:
            text += para.text
    return text


def get_txt_text(txt_docs):
    """
    Extracts text from TXT files.
    """
    text = ""
    for txt in txt_docs:
        text += txt.getvalue().decode("utf-8")
    return text


def get_csv_text(csv_docs):
    """
    Extracts text from CSV files.
    """
    text = ""
    for csv_file in csv_docs:
        reader = csv.reader(csv_file)
        for row in reader:
            text += " ".join(row) + "\n"
    return text


def get_xls_text(xls_docs):
    """
    Extracts text from XLS files.
    """
    text = ""
    for xls_file in xls_docs:
        wb = xlrd.open_workbook(file_contents=xls_file.getvalue())
        for sheet in wb.sheets():
            for row in range(sheet.nrows):
                text += " ".join([str(sheet.cell_value(row, col)) for col in range(sheet.ncols)]) + "\n"
    return text


def get_html_text(html_docs):
    """
    Extracts text from HTML files.
    """
    text = ""
    for html in html_docs:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html.getvalue(), 'html.parser')
        text += soup.get_text()
    return text


def get_text_chunks(text):
    """
    Splits text into smaller chunks for processing.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200, length_function=len)
    return text_splitter.split_text(text)


def get_vector_store(text_chunks):
    """
    Generates a vector store from text chunks.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain(model_name):
    """
    Returns a conversational chain for the selected model.
    """
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. 
    If the answer is not in the provided context, just say, "answer is not available in the context". 
    Context:\n {context}\n
    Question:\n {question}\n
    Answer:
    """
    
    model_mapping = {
        "Gemma 2": OllamaLLM(model="gemma2:9b"),
        "Llama 3.1": OllamaLLM(model="llama3.1:latest"),
        "Mistral": OllamaLLM(model="mistral:7b"),
        "Qwen 2": OllamaLLM(model="qwen2:7b")
    }
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model_mapping[model_name], chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question, model_name):
    """
    Handles user questions and retrieves a response using the selected model.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain(model_name)
    start_time = time.time()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    elapsed_time = time.time() - start_time
    return response["output_text"], elapsed_time


# ===========================
# Main Application
# ===========================

def main():
    # Set up the page
    st.set_page_config(page_title="AI Model Document Chat", page_icon="💬", layout="wide")

    # Header
    st.title("📄 AI Model Document Chat")
    st.subheader("Interact with your documents using advanced AI models.")

    # Tabs for content separation
    tab1, tab2 = st.tabs(["📂 Upload & Process", "💬 Chat"])

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = {}

    with tab1:
        st.header("📂 Upload Files")
        uploaded_files = st.file_uploader(
            "📂 Upload your files",
            accept_multiple_files=True,
            type=["pdf", "docx", "txt", "csv", "html", "xls"]
        )
        url_input = st.text_area("🌐 Enter URLs (comma-separated)", placeholder="https://example.com/doc.pdf, https://example.com/file.docx")

        if st.button("🚀 Submit & Process"):
            with st.spinner("Processing uploaded files..."):
                files_to_process = {}

                # Process uploaded files
                if uploaded_files:
                    for uploaded_file in uploaded_files:
                        file_extension = uploaded_file.name.split(".")[-1]
                        text = get_text_from_file(uploaded_file, file_extension)
                        files_to_process[uploaded_file.name] = text

                # Process files from URLs
                if url_input.strip():
                    fetched_files, url_errors = fetch_files_from_urls(url_input)
                    for file in fetched_files:
                        with open(file, "rb") as f:
                            file_extension = file.split(".")[-1]
                            text = get_text_from_file(f, file_extension)
                            files_to_process[file] = text
                    if url_errors:
                        for url, error in url_errors:
                            st.error(f"Failed to fetch file from {url}: {error}")

                if files_to_process:
                    st.session_state.processed_files = files_to_process
                    all_text = " ".join(files_to_process.values())
                    text_chunks = get_text_chunks(all_text)
                    get_vector_store(text_chunks)
                    st.success("✅ Files processed successfully!")
                else:
                    st.warning("⚠️ No valid files uploaded or processed.")

    with tab2:
        st.header("💬 Chat with Documents")

        if st.session_state.processed_files:
            # Allow user to select a file
            selected_file = st.selectbox(
            "📂 Select a file to query:",
            st.session_state.processed_files.keys()
        )
        user_question = st.text_input("❓ Ask a Question")
        model_name = st.selectbox("🤖 Select AI Model", ["Gemma 2", "Llama 3.1", "Mistral", "Qwen 2"])

        if st.button("💬 Get Response"):
            if not user_question.strip():
                st.warning("⚠️ Please enter a question!")
            else:
                with st.spinner(f"Getting response from {model_name}..."):
                    # Query the selected file's content
                    file_content = st.session_state.processed_files[selected_file]
                    text_chunks = get_text_chunks(file_content)
                    get_vector_store(text_chunks)
                    response, elapsed_time = user_input(user_question, model_name)

                    # Display model response
                    st.markdown(f"### {model_name} Reply:")
                    st.write(response)
                    st.write(f"⏳ Response Time: {elapsed_time:.2f} seconds")

                    # Add to chat history
                    st.session_state.chat_history.append({
                        "file": selected_file,
                        "question": user_question,
                        "response": response,
                        "time": elapsed_time
                    })

        # Filter chat history by the selected file
        file_specific_history = [
            chat for chat in st.session_state.chat_history if chat["file"] == selected_file
        ]

        # Check if there is any chat history for the selected file
        if len(file_specific_history) > 0:
            st.subheader(f"🕒 Chat History")

            # Generate the download file name based on the selected file
            base_file_name = selected_file.rsplit(".", 1)[0]  # Remove the .pdf extension
            download_file_name = f"{base_file_name}.txt"

            # Convert chat history to text for download
            chat_history_text = "\n".join(
            f"Q: {chat['question']}\nA: {chat['response']}\n"
            for chat in file_specific_history
            )

            st.download_button(label="Download Chat History", data=chat_history_text, file_name=download_file_name, mime="text/plain")
            for idx, chat in enumerate(file_specific_history):
                # Use the question text as the title of each chat in the expander
                with st.expander(f"Q: {chat['question']}"):
                     st.write(f"**Response:** {chat['response']}")
                     st.write(f"**Time Taken:** {chat['time']} seconds")
        else:
                # Display fallback message only when no history exists for the selected file
                if len(st.session_state.processed_files) > 0:
                    st.info(f"No chat history available for {selected_file}.")




if __name__ == "__main__":
    main()
