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

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_docx_text(docx_docs):
    text = ""
    for docx in docx_docs:
        doc = Document(docx)
        for para in doc.paragraphs:
            text += para.text
    return text

def get_txt_text(txt_docs):
    text = ""
    for txt in txt_docs:
        text += txt.getvalue().decode("utf-8")
    return text

def get_csv_text(csv_docs):
    text = ""
    for csv_file in csv_docs:
        reader = csv.reader(csv_file)
        for row in reader:
            text += " ".join(row) + "\n"
    return text

def get_xls_text(xls_docs):
    text = ""
    for xls_file in xls_docs:
        wb = xlrd.open_workbook(file_contents=xls_file.getvalue())
        for sheet in wb.sheets():
            for row in range(sheet.nrows):
                text += " ".join([str(sheet.cell_value(row, col)) for col in range(sheet.ncols)]) + "\n"
    return text

def get_html_text(html_docs):
    text = ""
    for html in html_docs:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html.getvalue(), 'html.parser')
        text += soup.get_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200, length_function=len)
    return text_splitter.split_text(text)
    pass

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    pass

def get_conversational_chain(model_name):
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. 
    If the answer is not in the provided context, just say, "answer is not available in the context". 
    Context:\n {context}\n
    Question:\n {question}\n
    Answer:
    """
    
    model_mapping = {
        "Gemma 2": OllamaLLM(model="gemma2:9b"),
        "Llama 3.1": OllamaLLM(model="llama3.1:8b"),
        "Mistral": OllamaLLM(model="mistral:7b"),
        "Qwen 2": OllamaLLM(model="qwen2:7b")
    }
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model_mapping[model_name], chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, model_name):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain(model_name)
    start_time = time.time()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    elapsed_time = time.time() - start_time
    return response["output_text"], elapsed_time

def fetch_files_from_urls(urls):
    """
    Fetches files from a list of URLs separated by commas and returns a list of file paths.
    """
    file_paths = []
    errors = []
    url_list = [url.strip() for url in urls.split(",") if url.strip()]

    for url in url_list:
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an error for bad responses
            file_name = url.split("/")[-1]  # Extract file name from URL
            if not any(file_name.endswith(ext) for ext in ['.pdf', '.docx', '.doc', '.txt', '.csv', '.html', '.xls']):
                errors.append(f"{url}: Unsupported file format.")
                continue

            temp_path = f"temp_{int(time.time())}_{file_name}"
            with open(temp_path, "wb") as f:
                f.write(response.content)
            file_paths.append(temp_path)
        except Exception as e:
            errors.append(f"{url}: Error fetching file - {e}")

    return file_paths, errors

def main():
    # Set up the page
    st.set_page_config(page_title="AI Model Document Chat", page_icon="💬", layout="wide")
    
    # Header
    st.title("📄 AI Model Document Chat")
    st.subheader("Interact with your documents using advanced AI models.")
    
    # Sidebar for navigation
    with st.sidebar:
        st.header("📋 Menu")
        uploaded_files = st.file_uploader(
            "📂 Upload your files", 
            accept_multiple_files=True, 
            type=["pdf", "docx", "txt", "csv", "html", "xls"]
        )
        url_input = st.text_area("🌐 Enter URLs (comma-separated)", placeholder="https://example.com/doc.pdf, https://example.com/file.docx")
        process_button = st.button("🚀 Submit & Process")
    
    # Tabs for content separation
    tab1, tab2 = st.tabs(["📂 Upload & Process", "💬 Chat"])
    
    with tab1:
        st.header("Upload & Process Files")
        
        if process_button:
            files_to_process = []
            with st.spinner("Processing uploaded files..."):
                # Process uploaded files
                if uploaded_files:
                    for uploaded_file in uploaded_files:
                        file_extension = uploaded_file.name.split(".")[-1]
                        if file_extension == "pdf":
                            files_to_process.append(get_pdf_text([uploaded_file]))
                        elif file_extension == "docx":
                            files_to_process.append(get_docx_text([uploaded_file]))
                        elif file_extension == "txt":
                            files_to_process.append(get_txt_text([uploaded_file]))
                        elif file_extension == "csv":
                            files_to_process.append(get_csv_text([uploaded_file]))
                        elif file_extension == "xls":
                            files_to_process.append(get_xls_text([uploaded_file]))
                        elif file_extension == "html":
                            files_to_process.append(get_html_text([uploaded_file]))
                
                # Process URLs
                if url_input:
                    fetched_files, url_errors = fetch_files_from_urls(url_input)
                    if fetched_files:
                        for file in fetched_files:
                            with open(file, "rb") as f:
                                file_extension = file.split(".")[-1]
                                if file_extension == "pdf":
                                    files_to_process.append(get_pdf_text([f]))
                                elif file_extension == "docx":
                                    files_to_process.append(get_docx_text([f]))
                                elif file_extension == "txt":
                                    files_to_process.append(get_txt_text([f]))
                                elif file_extension == "csv":
                                    files_to_process.append(get_csv_text([f]))
                                elif file_extension == "xls":
                                    files_to_process.append(get_xls_text([f]))
                                elif file_extension == "html":
                                    files_to_process.append(get_html_text([f]))

                    # Display errors if any
                    if url_errors:
                        for error in url_errors:
                            st.error(error)

                if files_to_process:
                    raw_text = " ".join(files_to_process)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("✅ Files processed successfully!")
                else:
                    st.warning("⚠️ No valid files uploaded or processed.")
    
    with tab2:
        st.header("Chat with Documents")
        
        user_question = st.text_input("❓ Ask a Question")
        model_name = st.selectbox("🤖 Select AI Model", ["Gemma 2", "Llama 3.1", "Mistral", "Qwen 2"])
    
    if st.button("💬 Get Response"):
        # AI response and document preview
        response, elapsed_time = user_input(user_question, model_name)
        st.write(f"### {model_name} Reply:")
        st.write(response)
        st.write(f"⏳ Response Time: {elapsed_time:.2f} seconds")
        
        # Relevant document context
        st.markdown("### 🔍 Relevant Context")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        docs = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True).similarity_search(user_question)
        for doc in docs:
            with st.expander("Relevant Context from Documents"):
                st.write(doc.page_content)


    # Chat History
    with st.sidebar:
        st.header("🕒 Chat History")
        chat_history = []  # Placeholder, should be dynamically updated
        if chat_history:
            for idx, chat in enumerate(chat_history):
                with st.expander(f"Chat {idx + 1}"):
                    st.write(f"**Question:** {chat['question']}")
                    st.write(f"**Response:** {chat['response']}")
                    st.write(f"**Time Taken:** {chat['time']} seconds")


if __name__ == "__main__":
    main()
