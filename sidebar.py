import os
import base64
import docx
import pandas as pd
from pptx import Presentation
import tempfile
import requests
import streamlit as st

ALLOWED_EXTENSIONS = ['.pdf', '.docx', '.doc', '.txt', '.ppt', '.csv', '.html', '.xls', '.xlsx']

def get_pdf_display_string(pdf_file_path):
    with open(pdf_file_path, 'rb') as pdf_file:
        base64_pdf = base64.b64encode(pdf_file.read()).decode('utf-8')
    pdf_display_string = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="500" height="1000" type="application/pdf"></iframe>'
    return pdf_display_string

def get_docx_text(uploaded_file):
    doc = docx.Document(uploaded_file)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)

def get_saved_files_info():
    return []

def is_allowed_extension(file_name):
    return os.path.splitext(file_name)[1].lower() in ALLOWED_EXTENSIONS

def save_uploaded_file(uploaded_file):
    if is_allowed_extension(uploaded_file.name):
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            st.toast(f"Saved {uploaded_file.name} to temporary storage.")
            return temp_file.name, os.path.splitext(uploaded_file.name)[1].lower()
    else:
        st.error("Unsupported file format.", icon="🚨")

def upload_file_via_url(url):
    try:
        response = requests.get(url)
        file_ext = os.path.splitext(url)[1].lower()
        if response.status_code == 200 and file_ext in ALLOWED_EXTENSIONS:
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
                temp_file.write(response.content)
                st.toast(f"Saved document from URL to temporary storage.")
                return temp_file.name, file_ext
        else:
            st.error('Invalid URL or unsupported content type.', icon="🚨")
    except requests.RequestException as e:
        st.error('Failed to fetch document from URL.', icon="🚨")


def sidebar():

    saved_files_info = get_saved_files_info()
    with st.sidebar:
        documents_uploads = st.file_uploader("Upload documents", accept_multiple_files=True, help="Supported formats: pdf, docx, doc, txt, ppt, csv, html, xls, xlsx")
        if documents_uploads:
            for uploaded_file in documents_uploads:
                file_info = save_uploaded_file(uploaded_file)
                if file_info:
                    saved_files_info.append(file_info)

        st.markdown("***")
    
        documents_uploads_url = st.text_input("Upload documents via url", help="Supported formats: pdf, docx, doc, txt, ppt, csv, html, xls, xlsx")
        submit_button = st.button('Upload link')
        if submit_button and documents_uploads_url:
            file_info = upload_file_via_url(documents_uploads_url)
            if file_info:
                saved_files_info.append(file_info)
        
        if saved_files_info:
            for file_info in saved_files_info:
                file_path, file_ext = file_info
                if file_ext == ".pdf":
                    st.markdown(get_pdf_display_string(file_path), unsafe_allow_html=True)
                elif file_ext == ".txt":
                    with open(file_path, 'r') as file:
                        text_content = file.read()
                        st.text_area("", text_content, height=300)
                elif file_ext == ".docx" or file_ext == ".doc":
                    document_text = get_docx_text(file_path)
                    st.text_area("", document_text, height=300)
                elif file_ext == ".csv":
                    df = pd.read_csv(file_path)
                    st.dataframe(df)
                elif file_ext == ".xls" or file_ext == ".xlsx":
                    df = pd.read_excel(file_path)
                    st.dataframe(df)

        st.markdown("***")
        complete_button = st.button("Submit & Process", disabled=not (saved_files_info))

        if complete_button:
            return saved_files_info
        else:
            return None
