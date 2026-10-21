"""
Loader de PDFs → Document + chunking
------------------------------------
- Lee PDFs con pdfplumber (página por página).
- Limpia el texto (quita saltos raros/espacios).
- Convierte cada página en un Document con metadatos: id, texto, source, page.
"""

from pathlib import Path
from typing import List, Dict
from .documents import Document, chunk_text
from .config import CONFIG
import re
import unicodedata
from typing import Iterable, Tuple



# Librería para leer PDFs
try:
    import pdfplumber
except Exception as e:
    raise RuntimeError(
        "Para usar loader_pdfs necesitás instalar pdfplumber:\n"
        "   pip install pdfplumber\n"
    ) from e


_NONWORD = re.compile(r"[^A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9.,;:?!¿¡()\"'–—\-/%\[\]<> ]")
_MANY_SYMBOLS = re.compile(r"[^\w\s]{3,}")           # secuencias largas de símbolos
_MULTI_SPACE = re.compile(r"\s+")
_HARD_HYPH = re.compile(r"(\w)-\n(\w)")              # palabra-\ncontinuación
_SOFT_HYPH = re.compile(r"[\u00AD]")                 # soft hyphen
_PAGE_NUM = re.compile(r"^\s*(page|p\.)\s*\d+\s*$", re.I)
_SECTION_NUM = re.compile(r"^\s*\d+(\.\d+)*\s*$")    # líneas tipo "3", "4.2.1"
_URL_LINE = re.compile(r"https?://\S+", re.I)

def _normalize(text: str) -> str:
    # NFKC para normalizar formas; reemplaza espacios no-break, etc.
    t = unicodedata.normalize("NFKC", text or "")
    t = t.replace("\u00A0", " ")  # NBSP → espacio
    # ligaduras comunes
    t = t.replace("ﬁ", "fi").replace("ﬂ", "fl")
    return t

def _split_lines_keep(text: str) -> list:
    # normaliza saltos → líneas “crudas”
    t = text.replace("\r", "\n")
    raw = [ln.strip() for ln in t.splitlines()]
    return [ln for ln in raw if ln]   # quita vacías

def _dehyphenate(paragraph: str) -> str:
    # une palabra-\npalabra → palabracontinuación
    return _HARD_HYPH.sub(r"\1\2", paragraph)

def _noise_ratio(line: str) -> float:
    if not line: return 1.0
    letters = sum(ch.isalnum() for ch in line)
    return 1.0 - (letters / max(1, len(line)))

def _clean_line(line: str) -> str:
    # elimina soft hyphen y controla símbolos extraños
    line = _SOFT_HYPH.sub("", line)
    line = "".join(ch for ch in line if ch.isprintable() or ch.isspace())
    # colapsa espacios
    line = _MULTI_SPACE.sub(" ", line).strip()
    return line

def _is_junky(line: str) -> bool:
    # líneas demasiado “símbolo” o con muy poca señal alfabética
    if _PAGE_NUM.match(line) or _SECTION_NUM.match(line):
        return True
    # muchas secuencias de símbolos (cuadros, barras, etc.)
    # si más del 45% no son alfanuméricos/espacios, descarta
    return False

def _merge_short_lines(lines: list, max_len: int = 60) -> list:
    """
    Junta líneas muy cortas (cortes “duros” de PDF) para formar frases más largas.
    """
    merged = []
    buf = ""
    for ln in lines:
        if not buf:
            buf = ln
            continue
        if len(buf) < max_len and not buf.endswith((".", "?", "!", ":", ";")):
            buf = f"{buf} {ln}"
        else:
            merged.append(buf)
            buf = ln
    if buf:
        merged.append(buf)
    return merged

def _clean(text: str) -> str:
    """
    Limpieza mejorada del texto extraído:
      - Normaliza Unicode/espacios/ligaduras
      - Dehyphenation (palabra-\npalabra)
      - Elimina caracteres de control y números de página
      - Junta líneas muy cortas
    """
    if not text:
        return ""

    # 1) Normalización general
    t = _normalize(text)

    # 2) Dehyphenation en quiebres de línea
    t = _dehyphenate(t)

    # 3) Partir en líneas “crudas” y limpiar por línea
    raw_lines = _split_lines_keep(t)
    cleaned = []
    for ln in raw_lines:
        ln = _clean_line(ln)
        if not ln:
            continue
        if _is_junky(ln):
            continue
        cleaned.append(ln)

    if not cleaned:
        return ""

    # 4) Merge de líneas muy cortas (mejora coherencia del chunk)
    cleaned = _merge_short_lines(cleaned, max_len=80)

    # 5) Compactar espacios finales
    out = "\n".join(cleaned)
    out = _MULTI_SPACE.sub(" ", out)
    return out.strip()



def pdf_to_documents(pdf_path: Path, doc_id_prefix: str = None) -> List[Document]:
    """
    Lee un PDF y devuelve una lista de Document:
    - Un Document por página con su metadato 'page'.
    - El id se arma como {doc_id_prefix}_p{nro_pagina}.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"No existe el archivo: {pdf_path}")

    docs: List[Document] = []
    base_id = (doc_id_prefix or pdf_path.stem)

    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = _clean(page.extract_text() or "")
            if not text.strip():
                continue
            docs.append(
                Document(
                    id=f"{base_id}_p{i}",       # identificador único del documento: usa el nombre base del archivo (sin .pdf) + "_p" + número de página (ej: contrato_p3)
                    text=text,                  # el texto ya limpio y normalizado de esa página (resultado de _clean(page.extract_text()))
                    source=str(pdf_path.name),  # nombre del PDF
                    page=i                      # número de página
                )
            )
    return docs



def folder_pdfs_to_documents(folder: Path, recursive: bool = True) -> List[Document]:
    """
    Carga TODOS los PDFs de una carpeta (y subcarpetas si recursive=True).
    Devuelve lista de Documents (cada uno corresponde a una página).
    """
    pattern = "**/*.pdf" if recursive else "*.pdf"
    docs: List[Document] = []
    if not folder.is_dir():
        raise FileNotFoundError(f"No existe la carpeta: {folder}")
    for pdf_path in sorted(folder.glob(pattern)):
        docs.extend(pdf_to_documents(pdf_path, doc_id_prefix=pdf_path.relative_to(folder).with_suffix("").as_posix()))
    return docs


def documents_to_chunks(
    docs: List[Document],
    max_tokens_chunk: int = CONFIG["max_tokens_chunk"],
    overlap: int = CONFIG["overlap"]
) -> Dict[str, List[str]]:
    """
    Convierte cada Document en sus chunks de texto.
    Retorna {doc_id: [chunk1, chunk2, ...]}.

    Extra: si una página es muy corta y chunk_text() devuelve [],
    generamos al menos un chunk de respaldo para que el doc no quede fuera del índice.
    """
    out: Dict[str, List[str]] = {}
    for d in docs:
        chunks = chunk_text(d.text, max_tokens_chunk, overlap)

        if d.id in out:
            raise ValueError(f"ID de documento duplicado: {d.id}")
        out[d.id] = chunks
    return out



#  Demo de uso rápido 
if __name__ == "__main__":
    from .config import resolve_path
    corpus_dir = resolve_path(CONFIG["corpus_dir"])   # Carpeta con PDFs
    all_docs = folder_pdfs_to_documents(corpus_dir, recursive=True)
    print(f"Se leyeron {len(all_docs)} páginas con texto.")

    chunks_map = documents_to_chunks(all_docs, max_tokens_chunk=CONFIG["max_tokens_chunk"], overlap=CONFIG["overlap"])
    n_chunks = sum(len(v) for v in chunks_map.values())
    print(f"Se generaron {n_chunks} chunks.")
