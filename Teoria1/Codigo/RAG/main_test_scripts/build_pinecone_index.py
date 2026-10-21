import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raglib.config import CONFIG, resolve_path
from raglib.loader_pdfs import folder_pdfs_to_documents, documents_to_chunks
from raglib.vector_pinecone import PineconeSearcher


def main():
    docs = folder_pdfs_to_documents(resolve_path(CONFIG["corpus_dir"]))
    chunks = documents_to_chunks(docs)
    if not any(chunks.values()):
        raise ValueError("El corpus no contiene texto indexable.")
    searcher = PineconeSearcher()
    searcher.upsert_chunks(chunks, {d.id: {"source": d.source, "page": d.page} for d in docs})
    print(f"Upsert completado: {sum(map(len, chunks.values()))} chunks.")


if __name__ == "__main__":
    main()
