import os 
import pickle
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import tempfile


INDEX_PATH = "ai_data/faiss_index.pkl"
DOCS_PATH = "ai_data/docs.pkl"

os.makedirs("ai_data", exist_ok=True)

def _read_pdf(file) -> str:
    """Read uploaded PDF (streamlit UploadedFile or open file)"""
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(file)
        text = []
        for p in reader.pages:
            text.append(p.extract_text() or "")
        return "\n".join(text)
    except Exception as e:
        print("PDF read error:", e)
        return ""

def create_embeddings_from_files(uploaded_files, openai_api_key=None):
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key

    texts, metadatas = [], []
    for f in uploaded_files:
        name = getattr(f, "name", "uploaded")
        if name.lower().endswith('.pdf'):
            text = _read_pdf(f)
        else:
            f.seek(0)
            text = f.read().decode('utf-8')
        texts.append(text)
        metadatas.append({"filename": name, "source": name})

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = []
    for i, t in enumerate(texts):
        chunks = splitter.split_text(t)
        for idx, chunk in enumerate(chunks):
            docs.append(Document(
                page_content=chunk,
                metadata={"source": metadatas[i]["filename"], "chunk": idx}
            ))

    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    vectordb = FAISS.from_documents(docs, embeddings)

    # persist
    with open(INDEX_PATH, "wb") as f:
        pickle.dump(vectordb, f)
    with open(DOCS_PATH, "wb") as f:
        pickle.dump(docs, f)

    return True

def load_vectorstore():
    if not os.path.exists(INDEX_PATH):
        return None
    with open(INDEX_PATH, "rb") as f:
        vectordb = pickle.load(f)
    return vectordb

def retrieve(query, k=4, openai_api_key=None):
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    vectordb = load_vectorstore()
    if not vectordb:
        return []
    return vectordb.similarity_search(query, k=k)
