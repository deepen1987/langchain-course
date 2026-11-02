from dotenv import load_dotenv
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import CharacterTextSplitter
# from langchain_ollama import OllamaEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import os

load_dotenv()

if __name__ == "__main__":
    print("Ingesting data...")
    loader = UnstructuredLoader(file_path="/Users/deependrashekhawat/Documents/GitHub/langchain-course/mediumblog1.txt", max_characters=1000000, chunking_strategy="basic", encoding="UTF-8")
    document = loader.load()

    print("Splitting")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"Length text {len(texts)} chunks")

    print("Embeddings")


    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview", output_dimensionality=1024)

    print("Ingesting")
    index_name = os.environ.get("INDEX_NAME")
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=1024,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    index = pc.Index(index_name)

    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    ids = [f"{os.path.basename(doc.metadata['source'])}#{i}" for i, doc in enumerate(texts)]
    vector_store.add_documents(documents=texts, ids=ids)