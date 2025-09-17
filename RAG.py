from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document


class RAGPipeline:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.vectorstore = None

    def build_index(self, text: str, source: str = "webpage"):
        """Split text, embed, and build FAISS index."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        docs = [
            Document(page_content=chunk, metadata={"source": source})
            for chunk in splitter.split_text(text)
        ]
        self.vectorstore = FAISS.from_documents(docs, self.embeddings)

    def query(self, query: str, k: int = 5):
        """Retrieve top-k chunks relevant to query."""
        if not self.vectorstore:
            raise ValueError("Vectorstore not built. Call build_index() first.")
        return self.vectorstore.similarity_search(query, k=k)

    def get_context(self, query: str, k: int = 5, max_chars: int = 4000) -> str:
        """Return concatenated top-k results as context for LLM."""
        docs = self.query(query, k=k)
        context = []
        total = 0
        for i, d in enumerate(docs):
            snippet = d.page_content.strip()
            label = d.metadata.get("source", "unknown")
            part = f"[[DOC {i} | source={label}]]\n{snippet}\n"
            total += len(part)
            if total > max_chars:
                break
            context.append(part)
        return "\n---\n".join(context)
