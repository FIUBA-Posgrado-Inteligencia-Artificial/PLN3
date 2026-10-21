from dataclasses import dataclass
from typing import List, Optional
import re
import unicodedata
from .config import CONFIG

@dataclass
class Document:
    id: str
    text: str
    source: str = ""
    page: Optional[int] = None

# tokenización básica (casefold + letras con acentos y dígitos) 
_TOKEN_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+")

def simple_tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())

# utilidades para chunking basado en oraciones 
# divide en oraciones simples: punto/exclamación/interrogación + espacio + mayúscula
_SENT_SPLIT = re.compile(r"(?<=[\.\!\?])\s+(?=[A-ZÁÉÍÓÚÑ])")
_SOFT_HYPH = re.compile(r"[\u00AD]")               # soft hyphen
_HARD_HYPH = re.compile(r"(\w)-\n(\w)")            # palabra-\ncontinuación

def _normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "")
    t = t.replace("\u00A0", " ")  # NBSP → espacio
    # de-hyphenation
    t = _SOFT_HYPH.sub("", t)
    t = _HARD_HYPH.sub(r"\1\2", t)
    # compacta espacios
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _split_sentences(text: str) -> List[str]:
    if not text:
        return []
    # no asumimos que todo empiece en mayúscula; si no hay cortes, devolvemos el texto entero
    parts = _SENT_SPLIT.split(text.strip())
    if len(parts) <= 1:
        return [text.strip()]
    # limpiar espacios
    return [" ".join(p.split()) for p in parts if p.strip()]

def _count_tokens(t: str) -> int:
    return len(simple_tokenize(t))

def _slide_merge(sents: List[str], max_tok: int, overlap: int) -> List[str]:
    text = " ".join(sents)
    tokens = list(_TOKEN_RE.finditer(text))
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tok, len(tokens))
        left = 0 if start == 0 else tokens[start].start()
        right = tokens[end].start() if end < len(tokens) else len(text)
        chunks.append(text[left:right].strip())
        if end == len(tokens):
            break
        start = end - overlap
    return chunks


def chunk_text(text: str, max_tokens: int = CONFIG["max_tokens_chunk"],
               overlap: int = CONFIG["overlap"]) -> List[str]:
    """Divide texto normalizado en ventanas con solapamiento de tokens léxicos."""
    if max_tokens <= 0 or not 0 <= overlap < max_tokens:
        raise ValueError("Se requiere max_tokens > 0 y 0 <= overlap < max_tokens.")
    return _slide_merge(_split_sentences(_normalize(text)), max_tokens, overlap)
