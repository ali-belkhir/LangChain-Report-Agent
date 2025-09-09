# RAG.py
import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# ---------- Persistent vector DB config ----------
CHROMA_DIR = os.environ.get("CHROMA_DIR", "chroma_db")
COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION", "rag_corpus")

_embeddings = OllamaEmbeddings(model="nomic-embed-text")
_vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=CHROMA_DIR,
    embedding_function=_embeddings,
)

_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""],
)

# ----------- Helpers -----------
def _fetch_url_text(url: str) -> str:
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for s in soup(["script", "style", "noscript"]):
        s.decompose()
    return soup.get_text(separator=" ", strip=True)

# ----------- Indexing -----------
def index_url(url: str) -> str:
    """Fetch content from a URL, chunk, and add to the vector index."""
    try:
        text = _fetch_url_text(url)
        if not text or len(text) < 50:
            return f"Fetched but little content at {url}"
        docs = [Document(page_content=chunk, metadata={"source": url})
                for chunk in _text_splitter.split_text(text)]
        _vectorstore.add_documents(docs)
        _vectorstore.persist()
        return f"Indexed {len(docs)} chunks from {url}"
    except Exception as e:
        return f"Error indexing {url}: {e}"

def index_csv(file_path: str) -> str:
    """Load a CSV and index its rows into the vector DB."""
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            return f"CSV at {file_path} is empty"
        rows_as_text = df.astype(str).apply(
            lambda r: " | ".join(f"{c}: {r[c]}" for c in df.columns), axis=1
        ).tolist()
        docs = [Document(page_content=chunk, metadata={"source": file_path})
                for chunk in _text_splitter.split_text("\n".join(rows_as_text))]
        _vectorstore.add_documents(docs)
        _vectorstore.persist()
        return f"Indexed {len(docs)} chunks from {file_path}"
    except Exception as e:
        return f"Error indexing CSV {file_path}: {e}"

# ----------- Retrieval -----------
def retrieve_relevant(query: str, k: int = 5) -> list[str]:
    """Retrieve top-k most relevant chunks for a query."""
    results = _vectorstore.similarity_search(query, k=k)
    return [doc.page_content for doc in results]
