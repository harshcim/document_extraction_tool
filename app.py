import os
import sys
import streamlit as st # type: ignore

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from scripts.embedding_creation import load_data, split_documents, save_embeddings # type: ignore

# from embedding_creation import load_data, split_documents, save_embeddings # type: ignore

from scripts.document_extraction import load_embedding_data, create_pipeline, extract_documents # type: ignore

# from document_extraction import load_embedding_data, create_pipeline, extract_documents # type: ignore

# Streamlit App UI
st.title("Document Extraction Tool")
st.write("Upload your PDF files and extract required tender documents.")

# Upload PDF files
uploaded_files = st.file_uploader("Upload PDF Documents", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    # Save uploaded files to data/ folder
    os.makedirs("data/", exist_ok=True)
    for file in uploaded_files:
        with open(os.path.join("data/", file.name), "wb") as f:
            f.write(file.read())
    st.success("PDF files uploaded successfully.")

    # Trigger the embedding creation
    if st.button("Process Documents and Create Embeddings"):
        documents = load_data("data/")
        documents_split = split_documents(documents)
        save_embeddings(documents_split, "vector_store/")
        st.success("Embeddings created successfully.")

    # Perform document extraction
    if st.button("Extract Required Documents"):
        context = st.text_area("Enter Context", value="Provide the list of documents required for tender filling.")
        if context:
            db = load_embedding_data("vector_store/")
            llm = create_pipeline()
            result = extract_documents(context, db, llm)
            st.write("Extracted Documents:")
            st.write(result)
