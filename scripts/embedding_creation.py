from langchain.text_splitter import RecursiveCharacterTextSplitter # type: ignore
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader # type: ignore
from langchain_community.vectorstores import FAISS # type: ignore
from langchain_google_genai import GoogleGenerativeAIEmbeddings # type: ignore
import google.generativeai as genai # type: ignore
import os
from dotenv import load_dotenv # type: ignore

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load and process PDF files
def load_data(directory):
    loader = DirectoryLoader(directory, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

# Split documents into chunks
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=150)
    return splitter.split_documents(documents)

# Save embeddings to FAISS
def save_embeddings(documents, output_dir):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    db = FAISS.from_documents(documents, embeddings)
    db.save_local(output_dir)

if __name__ == "__main__":
    documents = load_data("data/")
    documents_split = split_documents(documents)
    save_embeddings(documents_split, "vector_store/")
