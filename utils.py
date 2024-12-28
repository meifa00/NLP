import os
import streamlit as st
from langchain.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    UnstructuredHTMLLoader,
    UnstructuredExcelLoader,
)
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.llms import HuggingFaceHub
from sentence_transformers import SentenceTransformer
from langchain_ollama.llms import OllamaLLM
import langid
from dotenv import load_dotenv
import deepl

load_dotenv()
DEEPL_API_KEY = os.getenv("DEEPL_API_KEY")
translator = deepl.Translator(DEEPL_API_KEY)

def load_document(file_path, file_ext):
    if file_ext == '.pdf':
        return PyPDFLoader(file_path=file_path).load()
    elif file_ext == '.txt':
        return TextLoader(file_path=file_path).load()
    elif file_ext in ['.doc', '.docx']:
        return Docx2txtLoader(file_path=file_path).load()
    elif file_ext == ['.ppt', '.pptx']:
        return UnstructuredPowerPointLoader(file_path=file_path).load()
    elif file_ext == '.html':
        return UnstructuredHTMLLoader(file_path=file_path).load()
    elif file_ext in ['.xls', '.xlsx']:
        return UnstructuredExcelLoader(file_path=file_path).load()
    elif file_ext == '.csv':
        return CSVLoader(file_path=file_path).load()

def load_multiple_documents(file_infos):
    documents_to_text = []
    for file_path, file_ext in file_infos:
        documents_to_text.extend(load_document(file_path, file_ext))
    return documents_to_text

# def split_documents(documents):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
#     text_chunks = text_splitter.split_documents(documents)
#     return text_chunks

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    text_chunks = []
    for document in documents:
        text_content = document.page_content
        text_chunks.extend(text_splitter.split_text(text_content))
    return text_chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store

def create_conversational_chain(vector_store):
    template = """
    As a highly competent AI assistant, your role is analyze the documents provided by the user, Your responses should be grounded in the context contained within these documents.
      For each user query provided in the form of chat, apply the following guidelines:
    - If the answer is within the document's context, provide a detailed and precise response.
    - If the answer is not available based on the given context, clearly state that you don't have the information to answer.
    - If the query falls outside the scope of the context, politely clarify that your expertise is limited to answering questions directly related to the provided documents.

    When responding, prioritize accuracy and relevance:
    Context:
    {context}

    Question:
    {question}
    """
    system_message_prompt = SystemMessagePromptTemplate.from_template(template=template)
    human_template = "{question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    qa_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    memory = ConversationBufferMemory(memory_key='chat_history',return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(llm=OllamaLLM(model="llama3.1:8b"), retriever=vector_store.as_retriever(), memory=memory, combine_docs_chain_kwargs={'prompt': qa_prompt})
    return qa_chain

def initialize_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How can I help?"}]
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None

def load_qa_chain(saved_files_info):
    loaded_docs = load_multiple_documents(saved_files_info)
    docs_splits = split_documents(loaded_docs)
    vectordb = get_vectorstore(docs_splits)
    return create_conversational_chain(vectordb)

def format_chat_history():
    chat_history = []
    for message in st.session_state.messages:
        role = "Assistant" if message["role"] == "assistant" else "User"
        chat_history.append(f"{role}: {message['content']}")
    return "\n".join(chat_history)

def detect_language(text):
    lang, _ = langid.classify(text)
    return lang

def translate_text(text, target_language):
    try:
        result = translator.translate_text(text, target_lang=target_language.upper())
        return result.text
    except deepl.DeepLException as e:
        return f"Error in translation: {str(e)}"