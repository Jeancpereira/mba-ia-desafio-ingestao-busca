import hashlib
import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf.errors import FileNotDecryptedError, PdfReadError

from providers import get_vector_store, require_env

load_dotenv()

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def _chunk_id(pdf_path, chunk):
    page = chunk.metadata.get("page", "")
    digest = hashlib.sha256(chunk.page_content.encode("utf-8")).hexdigest()
    return f"{pdf_path}:{page}:{digest}"


def ingest_pdf():
    pdf_path = require_env("PDF_PATH")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF não encontrado: {pdf_path}")

    print(f"Carregando PDF: {pdf_path}")
    try:
        documents = PyPDFLoader(pdf_path).load()
    except FileNotDecryptedError as error:
        raise ValueError(
            f"PDF protegido por senha, não é possível ler: {pdf_path}"
        ) from error
    except PdfReadError as error:
        raise ValueError(f"PDF corrompido ou ilegível: {pdf_path} ({error})") from error
    print(f"Páginas carregadas: {len(documents)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    print(f"Chunks gerados: {len(chunks)}")

    if not chunks:
        raise ValueError(
            "Nenhum chunk gerado a partir do PDF — o arquivo pode estar vazio "
            "ou ser um PDF escaneado sem texto extraível (sem OCR)."
        )

    ids = [_chunk_id(pdf_path, chunk) for chunk in chunks]

    store = get_vector_store()
    store.add_documents(chunks, ids=ids)
    print("Ingestão concluída com sucesso.")


if __name__ == "__main__":
    ingest_pdf()
