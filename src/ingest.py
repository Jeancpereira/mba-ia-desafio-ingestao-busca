import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from providers import get_vector_store

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def ingest_pdf():
    if not PDF_PATH or not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF não encontrado: {PDF_PATH}")

    print(f"Carregando PDF: {PDF_PATH}")
    documents = PyPDFLoader(PDF_PATH).load()
    print(f"Páginas carregadas: {len(documents)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    print(f"Chunks gerados: {len(chunks)}")

    store = get_vector_store()
    store.add_documents(chunks)
    print("Ingestão concluída com sucesso.")


if __name__ == "__main__":
    ingest_pdf()
