import os
import shutil
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

DOCUMENTS_FOLDER = Path(os.getenv("DOCUMENTS_FOLDER", PROJECT_ROOT / "documents"))
CHROMA_DB_PATH = Path(os.getenv("CHROMA_DB_PATH", PROJECT_ROOT / "chroma_db"))


def _load_db() -> Chroma:
    return Chroma(
        persist_directory=str(CHROMA_DB_PATH),
        embedding_function=OpenAIEmbeddings(),
    )


def inspect_vector_store(query: str | None = None, limit: int = 5):
    """Print indexed source files and optional search preview with metadata."""
    if not CHROMA_DB_PATH.exists():
        print(f"No vector store found at: {CHROMA_DB_PATH}")
        print("Run: python -m rag_crew.vector_store")
        return

    db = _load_db()
    data = db.get(include=["documents", "metadatas"])
    total = len(data["ids"])
    sources = Counter(meta.get("source", "unknown") for meta in data["metadatas"])

    print(f"Vector store: {CHROMA_DB_PATH}")
    print(f"Total chunks: {total}\n")
    print("Indexed files:")
    for source, count in sorted(sources.items(), key=lambda item: item[0].lower()):
        print(f"  [{count:4d} chunks] {source}")

    if query:
        print(f"\nTop {limit} matches for: {query!r}\n")
        results = db.similarity_search(query, k=limit)
        for index, doc in enumerate(results, start=1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            preview = doc.page_content[:200].replace("\n", " ")
            print(f"{index}. {Path(source).name} (page {page})")
            print(f"   {preview}...\n")


def build_vector_store(docs_folder: str | None = None):
    folder = Path(docs_folder) if docs_folder else DOCUMENTS_FOLDER

    loader = PyPDFDirectoryLoader(
        str(folder),
        glob="**/*.pdf",
        recursive=True,
    )
    docs = loader.load()

    if not docs:
        print(f"No PDF files found in '{folder}'. Add PDFs to that folder and try again.")
        return None

    if CHROMA_DB_PATH.exists():
        shutil.rmtree(CHROMA_DB_PATH)
        print(f"Cleared old vector store at: {CHROMA_DB_PATH}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=str(CHROMA_DB_PATH),
    )

    print(f"Reading from: {folder}")
    print(f"Loaded {len(docs)} document(s)")
    print(f"Vector store saved to: {CHROMA_DB_PATH}")
    print(f"Created {len(chunks)} chunks.")
    return db


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "inspect":
        search_query = sys.argv[2] if len(sys.argv) > 2 else None
        inspect_vector_store(query=search_query)
    else:
        build_vector_store()
