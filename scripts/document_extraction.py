from langchain_community.vectorstores import FAISS # type: ignore
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI # type: ignore
from langchain.prompts import PromptTemplate # type: ignore
from langchain.chains import RetrievalQA # type: ignore
import google.generativeai as genai # type: ignore
from dotenv import load_dotenv # type: ignore
import os

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load embeddings from FAISS
def load_embedding_data(vector_store_dir):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    db = FAISS.load_local(vector_store_dir, embeddings, allow_dangerous_deserialization=True)
    return db

# Create the LLM pipeline
def create_pipeline():
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.5, max_output_tokens=4096)
    return model

# Extract required documents based on the context
def extract_documents(context, db, llm):
    sys_prompt = """You are an advanced and intelligent document extraction system. I will provide 
    you with the context of a tender document, and you need to extract the list of required documents
    mentioned in it. Some example documents include 'Turn Over', 'Balance Sheet','Profit & Loss', 
    'Solvency', 'Income Tax Return', 'PAN Card', and 'GST'. However, there may be additional documents 
    implied in the context. Extract both explicitly mentioned and any relevant documents required for tender
    filling.
    
    List of Known Documents: [ "Turn Over", "Balance Sheet", "Profit & Loss", "Solvency", "Working Capital", 
    "Income Tax Return", "PAN Card", "GST", "EPFO", "ESIC", "Blacklist", "Annexure", "Declaration", "Undertaking",
    "License", "Registration", "Registered", "Schedule", "UDYAM", "MSME", "Certificate", "Certification", "ISO", 
    "Experience", "Criteria", "Financial", "Commercial", "Technical", "Pre-Qualification", "Qualification", "Bid", 
    "Capacity", "Work in Hand", "Similar Work", "Manufacturer Authorisation", "OEM Authorisation", "Manufacture Authorisation",
    "Power of Attorney", "Authorisation", "Affidavit", "Checklist", "Performance"]
    
    Output Format: List of extracted documents
    
    Context: \n{context}\n
    """

    prompt = PromptTemplate(template=sys_prompt, input_variables=["context"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 5}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    answer = qa_chain.invoke({"query": context})
    return answer['result']

if __name__ == "__main__":
    db = load_embedding_data("vector_store/")
    llm = create_pipeline()

    # Provide the context for testing
    context = "Provide the list of required documents for tender filling, including financial and legal documents."
    response = extract_documents(context, db, llm)
    print(response)
