from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import os
import time
from .config import CONFIG
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

# Convención de IDs de chunk: f"{doc_id}::chunk_{local_idx}"
def make_chunk_id(doc_id: str, local_idx: int) -> str:
    return f"{doc_id}::chunk_{local_idx}"

def parse_chunk_id(chunk_id: str) -> Tuple[str, int]:
    doc_id, rest = chunk_id.rsplit("::chunk_", 1)
    i = int(rest)
    return doc_id, i

def ensure_pinecone_index(
    pc: Pinecone,
    name: str,
    dim: int,
    cloud: str = CONFIG["cloud"],
    region: str = CONFIG["region"],
    metric: str = CONFIG["metric"],
):
    idxs = pc.list_indexes().names()
    if name not in idxs:
        pc.create_index(
            name=name,
            dimension=dim,
            metric=metric,
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
    deadline = time.monotonic() + CONFIG["index_ready_timeout"]
    while True:
        description = pc.describe_index(name)
        if description.dimension != dim or description.metric != metric:
            raise ValueError("La dimensión o métrica del índice no coincide con el modelo configurado.")
        if description.status["ready"]:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(f"El índice {name} no está disponible.")
        time.sleep(1)


@dataclass
class PineconeSearcher:
    """
    Maneja embeddings + upsert + query sobre Pinecone.
    """
    index_name: str = CONFIG["index_name"]
    model_name: str = CONFIG["model_name"]
    cloud: str = CONFIG["cloud"]
    region: str = CONFIG["region"]
    api_key: Optional[str] = None
    namespace: str = CONFIG["namespace"]

    def __post_init__(self):
        key = self.api_key or os.getenv("PINECONE_API_KEY")
        if not key:
            raise RuntimeError("Falta PINECONE_API_KEY en entorno o parámetro api_key.")
        self.pc = Pinecone(api_key=key)
        self.model = SentenceTransformer(self.model_name)
        dim = self.model.get_sentence_embedding_dimension()
        ensure_pinecone_index(self.pc, self.index_name, dim, cloud=self.cloud, region=self.region)
        self.index = self.pc.Index(self.index_name)

    def _ns_vector_count(self) -> int:
        """Cantidad de vectores en la namespace actual."""
        stats = self.index.describe_index_stats()
        node = stats.namespaces.get(self.namespace)
        return int(node.vector_count) if node else 0

    def clear_namespace(self):
        stats = self.index.describe_index_stats()
        if self.namespace in stats.namespaces:
            self.index.delete(delete_all=True, namespace=self.namespace)

    def upsert_chunks(self, chunks_per_doc: Dict[str, List[str]], docs_meta: Dict[str, Dict]):
        vectors = []
        for doc_id, chunks in chunks_per_doc.items():
            if not chunks:
                continue
            embs = self.model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
            for i, (ch, v) in enumerate(zip(chunks, embs)):
                chunk_id = make_chunk_id(doc_id, i)
                meta = {
                    "doc_id": doc_id,
                    "local_idx": i,
                    "source": (docs_meta.get(doc_id, {}) or {}).get("source", ""),
                    "page": (docs_meta.get(doc_id, {}) or {}).get("page", None),
                    "text": ch,
                }
                meta = {key: value for key, value in meta.items() if value is not None}
                vectors.append({"id": chunk_id, "values": v.tolist(), "metadata": meta})
                if len(vectors) >= CONFIG["upsert_batch_size"]:
                    self.index.upsert(vectors=vectors, namespace=self.namespace)
                    vectors.clear()

        if vectors:
            self.index.upsert(vectors=vectors, namespace=self.namespace)


    def search(self, query: str, top_k: int = CONFIG["top_retrieve"], meta_filter: Optional[dict] = None) -> List[Tuple[str, float, Dict]]:
        """
        Devuelve [(chunk_id, score, meta)], score mayor = más similar
        """
        q = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0].tolist()
        res = self.index.query(
            vector=q,
            top_k=top_k,
            include_metadata=True,
            namespace=self.namespace,     # usamos namespace
            filter=meta_filter,           # opcional: filtrar por source/page/etc.
        )
        out = []
        for m in res.matches:
            meta = m.metadata or {}
            out.append((m.id, float(m.score), meta))
        return out
