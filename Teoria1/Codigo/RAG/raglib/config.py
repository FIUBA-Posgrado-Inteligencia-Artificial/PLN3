import json
import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
with (PROJECT_ROOT / "config.json").open(encoding="utf-8") as f:
    CONFIG = json.load(f)
for key, variable in {
    "index_name": "PINECONE_INDEX", "cloud": "PINECONE_CLOUD",
    "region": "PINECONE_REGION", "namespace": "PINECONE_NAMESPACE",
    "model_name": "EMBED_MODEL", "ce_model": "CE_MODEL",
    "summary_model": "OPENAI_MODEL",
}.items():
    if os.getenv(variable):
        CONFIG[key] = os.environ[variable]


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path
