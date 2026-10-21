import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from raglib.config import CONFIG, resolve_path

from raglib.pipeline import RagPipeline
from raglib.metrics import precision_at_k, recall_at_k, ndcg_at_k, mrr
from raglib.io_utils import load_docs_jsonl, load_qrels_csv
from raglib.vector_pinecone import PineconeSearcher


def evaluate(pipeline: RagPipeline, qrels, ks=tuple(CONFIG["eval_ks"]), top_retrieve=CONFIG["top_retrieve"], top_final=CONFIG["eval_top_final"]):
    if not qrels or not ks or min(ks) <= 0 or top_final < max(ks) or top_retrieve < top_final:
        raise ValueError("Se requieren qrels y 0 < k <= top_final <= top_retrieve.")
    rows = []
    for query, rel_ids in qrels.items():
        # Recuperación híbrida (BM25 + vector) con metadatos
        cand = pipeline.retrieve_with_metadata(query, top_k=top_retrieve)

        # IDs de documento antes del re-ranqueo (dedup conserva orden)
        pre_ids = [doc_id for (doc_id, _chunk, _meta) in cand]
        pre_ids = list(dict.fromkeys(pre_ids))

        # Re-ranqueo con Cross-Encoder y deduplicación por documento
        rer = pipeline.reranker.rerank(query, cand)
        post_ids = [doc_id for (doc_id, _chunk, _meta, _score) in rer]
        post_ids = list(dict.fromkeys(post_ids))[:top_final]
        pre_ids = pre_ids[:top_final]

        # Métricas por cada K
        for k in ks:
            rows.append({
                "query": query, "k": k,
                "precision_pre":  precision_at_k(pre_ids,  rel_ids, k),
                "recall_pre":     recall_at_k(pre_ids,     rel_ids, k),
                "ndcg_pre":       ndcg_at_k(pre_ids,       rel_ids, k),
                "precision_post": precision_at_k(post_ids, rel_ids, k),
                "recall_post":    recall_at_k(post_ids,    rel_ids, k),
                "ndcg_post":      ndcg_at_k(post_ids,      rel_ids, k),
                "mrr_pre":        mrr(pre_ids,  rel_ids),
                "mrr_post":       mrr(post_ids, rel_ids),
            })
    df = pd.DataFrame(rows)
    agg = df.groupby("k").mean(numeric_only=True).reset_index()
    return df, agg


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", type=str, default="", help="data/docs_sample_en.jsonl")
    parser.add_argument("--qrels", type=str, default="", help="data/qrels_sample_en.csv (query,doc_id,label)")
    parser.add_argument("--ks", type=str, default=",".join(map(str, CONFIG["eval_ks"])))
    parser.add_argument("--top_retrieve", type=int, default=CONFIG["top_retrieve"])
    parser.add_argument("--top_final", type=int, default=CONFIG["eval_top_final"])
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    ks = tuple(int(x) for x in args.ks.split(",") if x.strip())

    # Resolver y verificar rutas
    docs_path  = resolve_path(args.docs)  if args.docs  else None
    qrels_path = resolve_path(args.qrels) if args.qrels else None

    if docs_path and qrels_path and docs_path.exists() and qrels_path.exists():
        print(f"[INFO] docs : {docs_path}")
        print(f"[INFO] qrels: {qrels_path}")
        docs  = load_docs_jsonl(docs_path)
        qrels = load_qrels_csv(qrels_path)
    else:
        raise SystemExit(
            f"[ERROR] No se encontraron archivos válidos.\n"
            f"  docs = {docs_path} (exists={docs_path.exists() if docs_path else None})\n"
            f"  qrels= {qrels_path} (exists={qrels_path.exists() if qrels_path else None})\n"
            f"  cwd  = {Path.cwd()}\n"
            f"Ejemplo: python -m main_test_scripts.evaluate_retrieval --docs data/docs_sample_en.jsonl --qrels data/qrels_sample_en.csv"
        )

    # PineconeSearcher con namespace aislada para la evaluación
    searcher = PineconeSearcher(namespace=CONFIG["eval_namespace"])

    # Construir pipeline
    pipeline = RagPipeline(
        docs=docs,
        pinecone_searcher=searcher,
        do_upsert=True,
        device=args.device
    )

    # Ejecutar evaluación
    df, agg = evaluate(pipeline, qrels, ks=ks, top_retrieve=args.top_retrieve, top_final=args.top_final)
    print("\n=== Métricas por query y K ===")
    print(df.to_string(index=False))
    print("\n=== Promedios (macro) por K ===")
    print(agg.to_string(index=False))
