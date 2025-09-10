from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# 1. Load web page
url = "https://tradingeconomics.com/"  # Example: commodities news
loader = WebBaseLoader(url)
docs = loader.load()

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# 3. Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. Store in vector DB
vectordb = Chroma.from_documents(chunks, embedding=embeddings, persist_directory="db_web")

# 5. Create retriever
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# 6. Load Ollama model
llm = Ollama(model="mistral")  # Or llama2, or any local Ollama model

# 7. Create Retrieval-QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# 8. Ask a question
query = "What is the inflation rate and GDP of United States ?"
result = qa_chain({"query": query})

print("Answer:", result["result"])
print("\nSources:", [doc.metadata['source'] for doc in result["source_documents"]])
