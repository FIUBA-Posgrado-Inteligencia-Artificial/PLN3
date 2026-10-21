import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raglib import RagPipeline, PineconeSearcher
from raglib.config import CONFIG, resolve_path
from raglib.loader_pdfs import folder_pdfs_to_documents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="Consulta sobre el corpus")
    args = parser.parse_args()
    docs = folder_pdfs_to_documents(resolve_path(CONFIG["corpus_dir"]))
    pipeline = RagPipeline(docs=docs, pinecone_searcher=PineconeSearcher())
    results = pipeline.retrieve_and_rerank(args.query)
    print("=== RESUMEN RAG ===")
    print(pipeline.build_summary_context(results))
    print("=== CONTEXTO CON CITAS ===")
    print(pipeline.build_cited_context(results))


if __name__ == "__main__":
    main()
