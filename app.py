import streamlit as st
from components.sidebar import sidebar
from components.utils import (
    initialize_state,
    load_qa_chain,
    format_chat_history,
    detect_language,
    translate_text
)

def main():
    # Set up the Streamlit page configuration and title
    st.set_page_config(page_title="ChatPDF", page_icon=":books:")
    st.title("A Mechanism for Extracting Information and Integrating Knowledge")
    
    # Invoke the sidebar for user inputs like file uploads
    saved_files_info = sidebar()
    st.markdown("***")
    st.subheader('Interaction with Documents')

    # Initialize the session state variables
    initialize_state()

    # Load the QA chain if documents
    if saved_files_info and not st.session_state.qa_chain:
        st.session_state.qa_chain = load_qa_chain(saved_files_info)

    # Enable the chat section regardless of the QA chain's state
    prompt = st.chat_input('Ask questions about the uploaded documents', key="chat_input")
    
    # Process user prompts and generate responses
    if prompt and (st.session_state.messages[-1]["content"] != prompt or st.session_state.messages[-1]["role"] != "user"):
        # Detect language of the input
        detected_language = detect_language(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("Retrieving relevant information and generating output..."):
            if not st.session_state.qa_chain:
                response = {
                    "zh": "我无法加载QA链。请确保上传了文档。",
                    "ms": "Saya tidak dapat memuatkan rantaian QA. Sila pastikan dokumen dimuat naik。",
                    "en": "I couldn't load the QA chain. Please ensure documents are uploaded."
                }.get(detected_language, "I couldn't load the QA chain. Please ensure documents are uploaded.")
            else:
                raw_response = st.session_state.qa_chain.run(prompt)
                response = (
                    translate_text(raw_response, detected_language)
                    if detected_language in ["zh", "ms"]
                    else raw_response
                )
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Display the conversation messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Export the chat history
    chat_history = format_chat_history()
    st.download_button(
        label="Export Chat History",
        data=chat_history,
        file_name="chat_history.txt",
        mime="text/plain",
    )

if __name__ == '__main__':
    main()
