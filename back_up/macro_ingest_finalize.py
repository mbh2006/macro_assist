from pathlib import Path
from dotenv import load_dotenv
import re
import os
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from transformers import AutoTokenizer
import subprocess
import tempfile
import shutil
import numpy as np
from rank_bm25 import BM25Okapi
import pickle
import json
from collections import defaultdict
import argparse
from sentence_transformers import SentenceTransformer

try:
    import chromadb
except ImportError:
    chromadb = None

try:
    from pypdf import PdfReader, PdfWriter
except ImportError:
    raise RuntimeError("PyPDF is required for internal layout splitting. Run: pip install pypdf")

try:
    from nltk.tokenize import sent_tokenize
except ImportError:
    sent_tokenize = None


# -------------- loading the directories and the embedding model settings -----------
load_dotenv()

# it moves to the main parent folder of my file:

#    ->macro_assist             =>parent[1]
#       ->ingest                =>parent[0]
#           macro_ingest.py

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PDF_DIR = os.getenv("MACRO_PDF_DIR", str(PROJECT_ROOT / "data" / "macro_textbooks"))
DEFAULT_PERSIST_DIR = os.getenv("MACRO_CHROMA_DIR", str(PROJECT_ROOT / "data" / "index" / "macro_chroma_db"))
DEFAULT_MANIFEST_PATH = os.getenv("MACRO_MANIFEST_PATH", str(PROJECT_ROOT / "data" / "index" / "macro_index_manifest.json"))
DEFAULT_BM25_PATH = os.getenv("MACRO_BM25_PATH", str(PROJECT_ROOT / "data" / "index" / "macro_bm25.pkl"))
DEFAULT_BM25_JSON_PATH = os.getenv("MACRO_BM25_JSON_PATH", str(PROJECT_ROOT / "data" / "index" / "macro_bm25.json"))
DEFAULT_EMBEDDING_MODEL = os.getenv("MACRO_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5" )
DEFAULT_COLLECTION = os.getenv("MACRO_COLLECTION_NAME", "macro_textbooks")
DEFAULT_EQUATION_INDEX_PATH = os.getenv("MACRO_EQUATION_INDEX_PATH", str(PROJECT_ROOT / "data" / "index" / "macro_equation_index.json"))

tokenizer = AutoTokenizer.from_pretrained(DEFAULT_EMBEDDING_MODEL)
# =================================================================

# --------------- helper functions for the equations --------------

# used to detect two flavors of equations
# 1). latex math blocks
# 2). textbook assignments
EQUATION_RE = re.compile(
    r"(\$\$.*?\$\$|\\\[.*?\\\]|\\\(.+?\\\)|^[A-Za-z][A-Za-z0-9_()\s\-]{0,60}\s*=\s*[^\n]{3,200}$)",
    re.DOTALL | re.MULTILINE,
    )

# inline equation extraction
INLINE_EQUATION_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9_]{0,20}\s*=\s*[^.,;\n]{2,120}")

# the raw text includes messy markup tags
# this function strips out those display boundaries
# it condenses any erratic multi-line spacing into:
# clean single spaces, and hands back a pure, readable formula string.
def normalize_equation(eq: str) -> str:
    eq = _normalize_text(eq)
    eq = eq.replace("$$", "").replace("\\[", "").replace("\\]", "")
    eq = eq.replace("\\(", "").replace("\\)", "")
    eq = re.sub(r"\s+", " ", eq).strip()
    return eq

def extract_equations_from_text(text: str) -> List[str]:
    if not text:
        return []
    
    hits = []
    # raw snippet that matches equation pattern, iterate through them one by one
    for m in EQUATION_RE.findall(text):
        # pass each raw match for cleaning in single line format
        eq = normalize_equation(m)
        # is it actually a math equation
        if "=" in eq or "\\" in eq:
            hits.append(eq)

    for m in INLINE_EQUATION_RE.findall(text):
        # pass each raw match for cleaning in single line format
        eq = normalize_equation(m)
        # is it actually a math equation
        if "=" in eq:
            hits.append(eq)

    # drops the duplicates while preserving the exact chronological reading order of testbook
    return list(dict.fromkeys(hits))

# metadata mapping matrix
def build_equation_index(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    equations = []
    for chunk in chunks:
        chunk_equations = chunk.get("equations", extract_equations_from_text(chunk.get("text", "")))
        for eq in chunk_equations:

            equations.append({
                "equation": eq,
                "chunk_id": chunk.get("chunk_id"),
                "book_title": chunk.get("book_title", ""),
                "chapter": chunk.get("chapter", ""),
                "section": chunk.get("section", ""),
                "source_path": chunk.get("source_path", ""),
                "chunk_type": chunk.get("chunk_type", ""),
            })

    return {"equations": equations}

# form a json file of equations
def save_equation_index_json(index: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

# =================================================================

# --------- helper functions for texts ----------------------------

def _normalize_text(text: str) -> str:
    # removes the junks of unicode 
    text = unicodedata.normalize("NFKC", text or "")
    # replace hidden, troble causing unicode characters into regular spaces
    text = text.replace("\u200b", " ").replace("\ufeff", " ")
    return re.sub(r"\s+", " ", text).strip()

# protected macro_terms for BM20 tokenization
_PROTECTED_TERMS = frozenset(
    {
        "is-lm", "ad-as", "phillips-curve", "solow-model", 
        "open-market", "money-multiplier", "gdp-deflator",
    }
    )

# tokenization for BM25
def tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    tokens = []
    for token in re.findall(r"[a-zA-Z0-9]+(?:[-'][a-zA-Z0-9]+)*", text):
        if token in _PROTECTED_TERMS:
            tokens.append(token)
        else:
            tokens.extend(re.findall(r"[a-zA-Z0-9]+", token))
    return tokens

# used to approximate token counts
def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))

# create clean ids used for chunk ids
def slugify(value: str) -> str:
    value = _normalize_text(value).lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "document"

# -------------- Chunking utility function -----------------------

def chunk_words(text: str, max_tokens: int = 250, overlap_tokens: int = 35) -> List[str]:
    words = (text or "").split()

    if not words:
        return []
    
    if len(words) <= max_tokens:
        return [" ".join(words).strip()]
    
    step = max(1, max_tokens - overlap_tokens)
    chunks = []

    for start in range(0, len(words), step):
        piece = words[start:start + max_tokens]
        if piece:
            chunks.append(" ".join(piece).strip())
        if start + max_tokens >= len(words):
            break

    return chunks

# do the sentence segmentation
def split_into_sentences(text: str) -> List[str]:
    text = _normalize_text(text)
    if not text:
        return []
    if sent_tokenize is not None:
        try:
            return [s.strip() for s in sent_tokenize(text) if s.strip()]
        except Exception:
            pass
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


# ----------- Structural Splitting -----------------------

# splits mineru markdown into sections using headers
def split_by_markdown_headers(markdown_text: str) -> List[str]:
    sections = re.split(r"(?=^#{1,6}\s)", markdown_text, flags=re.MULTILINE)

    cleaned = [s.strip() for s in sections if s.strip()]

    if cleaned:
        return cleaned
    
    paras = [p.strip() for p in re.split(r"\n\s*\n+", markdown_text or "") if p.strip()]
    return paras if paras else [markdown_text.strip()] if markdown_text.strip() else []

# to find section names
def extract_section_title(section: str) -> str:
    for line in section.splitlines():

        if re.match(r"^#{1,6}\s+", line):
            return re.sub(r"^#{1,6}\s+", "", line).strip()
        
    first_line = section.strip().splitlines()[0].strip() if section.strip() else ""
    return first_line[:120] if first_line else "Untitled Section"

# ---------- Math/Table routing ------------------------------

# detects equations, tables, is-lm, ad-as, gdp formulas
def is_math_or_table_heavy(text: str) -> bool:
    sample = text or ""
    markers = [
        r"\bIS\b", r"\bLM\b", r"\bAD\b", r"\bAS\b", r"=",
        r"\bGDP\b", r"\bCPI\b", r"\bPI\b", r"\bM/P\b",
        r"\\begin\{array\}", r"\\frac", r"\\sum", r"\\pi", r"\\Delta",
        "Table", "Figure", "|"
    ]
    hits = sum(1 for m in markers if re.search(m, sample, re.IGNORECASE))
    return hits >= 3

# ===============================================================

# ------------- Parsing with mineru -----------------------------

# automatically splits large files/books into 30-page chunks
# prevent mineru RAM crashes
# stitches markdown back together
# supports mathematical layouts

# convert pdf into markdown

def parse_with_mineru(file_path: str) -> str:
    
    # Uses MinerU via direct system CLI subprocesses for elite math parsing.
    # Natively auto-splits files into temporary sub-directories to prevent local RAM leaks.
    # Forces the stable 'pipeline' backend to avoid vllm/libcudart version compilation crashes.

    #  Dynamic Executable Auto-Discovery Gate to prevent crashing
    mineru_exe = shutil.which("mineru") or shutil.which("magic-pdf")
    if not mineru_exe:
        for fallback in ["/usr/local/bin/mineru", "/usr/local/bin/magic-pdf", "/root/.local/bin/mineru"]:
            if Path(fallback).exists():
                mineru_exe = fallback
                break
    if not mineru_exe:
        mineru_exe = "mineru"

    reader = PdfReader(file_path)
    total_pages = len(reader.pages)
    # pages to process at a time
    pages_per_chunk = 30  
    
    # If the document is small i.e., less than pages_per_chunk, pass it straight to MinerU
    if total_pages <= pages_per_chunk:
        print(f"[MinerU] Processing {Path(file_path).name} directly ({total_pages} pages) via {mineru_exe}...")

        # creating temporary files on your computer to store mineru's output files
        # after execution, python automatically delete those files
        with tempfile.TemporaryDirectory() as temp_out:
            # Appended '-b pipeline' to secure environment stability
            #  -p ->path to input file
            #  -o ->directory where extracted files saved
            #  -b ->instructs mineru to use its native pipeline
            cmd = [mineru_exe, "-p", file_path, "-o", temp_out, "-b", "pipeline"]

            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # handles processing failures
            if result.returncode != 0:
                raise RuntimeError(f"MinerU execution error: {result.stderr}")
            
            # validate the output content
            # it searches inside temporary directory for markdown files (.md)
            # if mineru said finich but no file produced it alerts the error
            md_files = list(Path(temp_out).rglob("*.md"))
            if not md_files:
                raise RuntimeError("MinerU finalized process successfully but no Markdown asset was generated.")
            
            # reads and cleans the extracted text
            with open(md_files[0], "r", encoding="utf-8") as f:
                return _normalize_text(f.read())

    print(f"[MinerU] Math-heavy layout detected ({total_pages} pages). Activating isolated sub-process loop...")
    # to store the text output of separate segment
    stitched_markdown_list = []

    # spun up a single temp dir to host all slice files and temp output logs safely
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        
        # sliding window looping
        for start_page in range(0, total_pages, pages_per_chunk):
            end_page = min(start_page + pages_per_chunk, total_pages)
            print(f"  -> [MinerU] Slicing and parsing section: pages {start_page + 1} to {end_page}...")
            
            # Step A: Slice textbook segment into a temporary PDF file
            writer = PdfWriter()
            for page_num in range(start_page, end_page):
                writer.add_page(reader.pages[page_num])
            # saves the extracted segment into a temp pdf chunk file  
            temp_chunk_path = temp_dir_path / f"mineru_chunk_{start_page}_{end_page}.pdf"
            with open(temp_chunk_path, "wb") as f:
                writer.write(f)
                
            # Step B: Establish an isolated chunk output directory for this runner
            chunk_output_dir = temp_dir_path / f"out_{start_page}_{end_page}"
            chunk_output_dir.mkdir(exist_ok=True)
            
            # Step C: Spin up an independent system thread forcing the pipeline layout
            # trigger mineru terminal on just the small file slice
            cmd = [mineru_exe, "-p", str(temp_chunk_path), "-o", str(chunk_output_dir), "-b", "pipeline"]
            
            # Capture execution safely
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"MinerU subprocess failure at boundary window {start_page}-{end_page}: {result.stderr}")
                
            # Step D: Locate the resulting Markdown string, load it, and free system memory hooks
            # in each unique chunk dir, opens it freshly extracted .md and reads the raw text and saves it into the storage list
            md_files = list(chunk_output_dir.rglob("*.md"))
            if not md_files:
                raise RuntimeError(f"MinerU completed execution pass, but could not locate the compiled markdown target data.")                
            with open(md_files[0], "r", encoding="utf-8") as f:
                stitched_markdown_list.append(f.read())
                
    full_book_markdown = "\n\n\n\n".join(stitched_markdown_list)
    print(f"Successfully compiled and healed full structural mathematical stream via MinerU for {Path(file_path).name}!")
    return _normalize_text(full_book_markdown)


def parse_pdf_to_markdown(file_path: str, parser_choice: str = "mineru") -> str:
    # Strictly routes layout payloads to the MinerU parser.
    parser_choice_norm = _normalize_text(parser_choice).lower()

    if parser_choice_norm != "mineru":
        raise RuntimeError(f"Unsupported parser choice: {parser_choice}. Only MinerU is enabled.")
    
    return parse_with_mineru(file_path)

# ==========================================================================================

# ------------------- Semantic chunking engine ---------------------------------------------

def semantic_chunking(
    structured_sections: Sequence[str],
    model: SentenceTransformer,
    similarity_threshold: float = 0.62,
    max_sentences_per_chunk: int = 20,
    max_tokens_per_chunk: int = 8000,
    overlap_sentences: int = 3,
) -> List[Dict[str, Any]]:
    
    # creates meaning aware chunks
    chunks = []
    chunk_id_counter = 1

    for section in structured_sections:
        section = section.strip()
        if not section:
            continue

        # cleans and strips headers
        section_title = extract_section_title(section)
        content_without_header = re.sub(r"^#{1,6}\s+.*?$", "", section, count=1, flags=re.MULTILINE).strip()

        # generate sentence embeddings
        sentences = split_into_sentences(content_without_header)
        if not sentences:
            continue

        # if small number of sentences in a section store the entire section as a single chunk
        if len(sentences) <= 2:
            chunk_text = f"Section: {section_title}\n\n" + " ".join(sentences)
            chunks.append({
                "chunk_id": chunk_id_counter,
                "section": section_title,
                "chapter": "",
                "book_title": "",
                "source_path": "",
                "chunk_type": "short_section",
                "token_count": count_tokens(chunk_text),
                "text": chunk_text,
            })
            chunk_id_counter += 1
            continue

        embeddings = model.encode(sentences, show_progress_bar=False, normalize_embeddings=True)
        embeddings = np.asarray(embeddings, dtype=np.float32)

        current_sentences = [sentences[0]]
        current_embeddings = [embeddings[0]]

        # now we do the math of the semantic embeddings
        for i in range(1, len(sentences)):
            # chunk_centroid represents the overall theme of the chunk at that moment
            chunk_centroid = np.mean(current_embeddings, axis=0)
            # find the magnitude of this chunk
            norm = np.linalg.norm(chunk_centroid)
            # makes a unit vector out of it
            if norm > 0:
                chunk_centroid = chunk_centroid / norm
            # calculate the cosine similarity by the dot product of unit vectors
            sim = float(np.dot(chunk_centroid, embeddings[i]))

            candidate_text = f"Section: {section_title}\n\n" + " ".join(current_sentences + [sentences[i]])
            candidate_tokens = count_tokens(candidate_text)

            # min threshold achived, within the max sentence limit, within max token count
            if (sim >= similarity_threshold        
                and len(current_sentences) < max_sentences_per_chunk
                and candidate_tokens <= max_tokens_per_chunk ):
                # append that sentence into the chunk
                current_sentences.append(sentences[i])
                current_embeddings.append(embeddings[i])
            else:
                # if failed closa the current chunk and make the new chunk so on
                chunk_text = f"Section: {section_title}\n\n" + " ".join(current_sentences)
                chunks.append({
                    "chunk_id": chunk_id_counter,
                    "section": section_title,
                    "chapter": "",
                    "book_title": "",
                    "source_path": "",
                    "chunk_type": "semantic_group",
                    "token_count": count_tokens(chunk_text),
                    "text": chunk_text,
                })
                chunk_id_counter += 1
                # enabling overlaping to prevent losing context due to boundary line
                overlap = current_sentences[-overlap_sentences:] if overlap_sentences > 0 else []
                overlap_emb = current_embeddings[-overlap_sentences:] if overlap_sentences > 0 else []
                current_sentences = overlap + [sentences[i]]
                current_embeddings = overlap_emb + [embeddings[i]]

        if current_sentences:
            chunk_text = f"Section: {section_title}\n\n" + " ".join(current_sentences)
            chunks.append({
                "chunk_id": chunk_id_counter,
                "section": section_title,
                "chapter": "",
                "book_title": "",
                "source_path": "",
                "chunk_type": "semantic_group",
                "token_count": count_tokens(chunk_text),
                "text": chunk_text,
            })
            chunk_id_counter += 1
    return chunks

# Fallback chunking, handels formula heavy content
def fixed_size_chunking(section_text: str, max_tokens: int = 240, overlap_tokens: int = 40) -> List[str]:
    section_text = _normalize_text(section_text)
    return chunk_words(section_text, max_tokens=max_tokens, overlap_tokens=overlap_tokens) if section_text else []

# ===========================================================================

# ---------------------- Chapter detection ----------------------------------

_CHAPTER_RE = re.compile(r"^#{1,2}\s*(?:Chapter|CHAPTER)\s+(\d+[\w\.]*.*?)$", re.MULTILINE)

# used to idenify the chapter ownership
# Finds nearest preceding chapter heading.
def extract_chapter_guess(markdown_text: str, section_start_pos: int) -> str:
    """Find the nearest preceding chapter heading before section_start_pos."""
    matches = list(_CHAPTER_RE.finditer(markdown_text))
    chapter = ""
    for m in matches:
        if m.start() <= section_start_pos:
            chapter = m.group(1).strip()
        else:
            break
    return chapter

# ==================================================================================

# --------------------------- Persistent layer -------------------------------------

# creates output directories safely
def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

# ==================================================================================

# ------------------- Sparse retrieval ---------------------------------------------

# >>>>>>>>>>>> sparse vectors <<<<<<<<<

# Build sparse retrieval index
def build_bm25_bundle(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a pickle-serializable BM25 bundle."""
    # extract the raw text
    corpus_texts = [chunk["text"] for chunk in chunks]
    # tokenize entire corpus
    tokenized_corpus = [tokenize(text) for text in corpus_texts]
    # build BM25 index mathematical engine
    bm25 = BM25Okapi(tokenized_corpus)

    return {
        "bm25": bm25,
        "chunks": chunks,
        "corpus_texts": corpus_texts,
        "tokenized_corpus": tokenized_corpus,
    }

# serialize bm25 index to macro_bm25.pkl
def save_bm25_bundle(bundle: Dict[str, Any], path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(bundle, f)

# save bm25 to json
def save_bm25_bundle_json(bundle: Dict[str, Any], path: str) -> None:
    # 1. Create a copy of the bundle dictionary
    json_safe_bundle = {
        "chunks": bundle["chunks"],
        "corpus_texts": bundle["corpus_texts"],
        "tokenized_corpus": bundle["tokenized_corpus"]
    }
    
    # 2. Save it to disk as a clean, formatted JSON file
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe_bundle, f, ensure_ascii=False, indent=2)


def load_bm25_bundle_json(path: str) -> Dict[str, Any]:
    # 1. Read the raw text data back from the JSON file
    with open(path, "r", encoding="utf-8") as f:
        bundle = json.load(f)
        
    # 2. Reconstruct the mathematical BM25 engine on the fly using the tokens
    bundle["bm25"] = BM25Okapi(bundle["tokenized_corpus"])
    
    return bundle

# ==================================================================================

# --------------- manifest generation --------------------------

# Stores: books, chunk counts, model name, parser, collection name
def save_manifest(
    manifest_path: str | Path,
    *,
    pdf_dir: str,
    persist_dir: str,
    collection_name: str,
    embedding_model_name: str,
    parser_choice: str,
    total_books: int,
    total_chunks: int,
    books_summary: List[Dict[str, Any]],
) -> None:
    
    manifest = {
        "pdf_dir": str(Path(pdf_dir).resolve()),
        "persist_dir": str(Path(persist_dir).resolve()),
        "collection_name": collection_name,
        "embedding_model_name": embedding_model_name,
        "parser_choice": parser_choice,
        "total_books": total_books,
        "total_chunks": total_chunks,
        "books_summary": books_summary,
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

# =========================================================================

# ----------------------- loading pdfs ---------------------------

def load_textbooks(pdf_dir: str) -> List[Path]:
    root = Path(pdf_dir)
    return sorted(root.rglob("*.pdf")) if root.exists() else []

# =========================================================================

# -------------------- Chroma db layer ----------------------------

# >>>>>>> Dense vectors <<<<<<<

# builds the dense vector data base
# The '*' forces all following parameters to be passed explicitly as keyword-only arguments.
def upsert_into_chroma(
    *,
    chunks: List[Dict[str, Any]],
    persist_dir: str,
    collection_name: str,
    embedding_model_name: str,
    rebuild: bool = True,
    batch_size: int = 128,
) -> None:
    
    if chromadb is None:
        raise RuntimeError("chromadb not installed")
    
    # establish local disk storage, create physical database folder on hard drive ar persist_dir tosave data permanently
    client = chromadb.PersistentClient(path=str(Path(persist_dir).resolve()))

    # if rebuild is true the function looks for any pre_existing database collection with same name and wipe it out
    if rebuild:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

    # prepares database collection and explicitly sets structural layout to use cosine similaritiy
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    embedding_model = SentenceTransformer(embedding_model_name)

    # chroma expects parallel lists rather than list of nested dictionaries
    ids = []            # list of text strings
    documents = []      # raw text blocks that are actually searched and read
    metadatas = []      # pure key-value tracling dictionaries

    # The function loops through your chunks and divides them into three separate arrays
    for chunk in chunks:
        ids.append(str(chunk["chunk_id"]))

        documents.append(chunk["text"])

        metadatas.append({
            "book_title": chunk.get("book_title", ""),
            "book_slug": chunk.get("book_slug", ""),
            "chapter": chunk.get("chapter", ""),
            "section": chunk.get("section", ""),
            "section_index": int(chunk.get("section_index", 0) or 0),
            "source_path": chunk.get("source_path", ""),
            "chunk_type": chunk.get("chunk_type", ""),
            "token_count": int(chunk.get("token_count", 0) or 0),
            "equation_count": len(chunk.get("equations", [])),
        })

    # handle batch embeddings and data upsert
    for start in range(0, len(chunks), batch_size):
        end = start + batch_size
        batch_ids = ids[start:end]
        batch_docs = documents[start:end]
        batch_metas = metadatas[start:end]

        # generate the multi-dimensional semantic vectors for just that specific batch block.
        batch_embeddings = embedding_model.encode(
            batch_docs,
            batch_size=min(32, len(batch_docs)),
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        # An upsert is a hybrid database command—if a chunk ID doesn't exist, it inserts it fresh; if the chunk ID already exists, it updates it with the new data.
        collection.upsert(ids=batch_ids, documents=batch_docs, metadatas=batch_metas, embeddings=batch_embeddings)

# =================================================================================

# ------------ Continuous book stitching engine -------------------

# chunk an already stitched book

def build_chunks_for_text_stream(
    markdown_text: str,
    book_title: str,
    master_source_path: str,
    embedding_model: SentenceTransformer,
    similarity_threshold: float = 0.62,
    semantic_max_sentences: int = 8,
    semantic_max_tokens: int = 420,
    overlap_sentences: int = 1,
    fixed_max_tokens: int = 240,
    fixed_overlap_tokens: int = 40,
) -> List[Dict[str, Any]]:
    
    """ Runs the hybrid semantic/fixed chunking over a stitched, continuous text block """

    # section extraction
    sections = split_by_markdown_headers(markdown_text)

    if not sections:
        sections = [markdown_text]

    book_slug = slugify(book_title)
    chunks = []
    chunk_global_id = 1
    current_chapter = ""

    for section_index, section in enumerate(sections, start=1):
        section = section.strip()
        # if section empty move on
        if not section:
            continue
        
        # Find position to preserve continuous chapter guessing
        section_start = markdown_text.find(section) if markdown_text else -1
        if section_start >= 0:
            current_chapter = extract_chapter_guess(markdown_text, section_start)

        section_title = extract_section_title(section)
        content = re.sub(r"^#{1,6}\s+.*?$", "", section, count=1, flags=re.MULTILINE).strip()
        content = content or section

        # Route math heavy structural content to fixed windows
        if is_math_or_table_heavy(content):
            for piece_index, piece in enumerate(
                fixed_size_chunking(content, max_tokens=fixed_max_tokens, overlap_tokens=fixed_overlap_tokens), 1
            ):
                text = f"Book: {book_title}\nSection: {section_title}\n\n{piece}"

                chunks.append({
                    "chunk_id": f"{book_slug}-{chunk_global_id}",
                    "book_title": book_title,
                    "book_slug": book_slug,
                    "chapter": current_chapter,
                    "section": section_title,
                    "section_index": section_index,
                    "source_path": master_source_path,
                    "chunk_type": "fixed_math_or_table",
                    "token_count": count_tokens(text),
                    "equations": extract_equations_from_text(text),
                    "text": text,
                })

                chunk_global_id += 1
            continue

        # Process textual sections with sentence-similarity models
        section_chunks = semantic_chunking(
            [section],
            model=embedding_model,
            similarity_threshold=similarity_threshold,
            max_sentences_per_chunk=semantic_max_sentences,
            max_tokens_per_chunk=semantic_max_tokens,
            overlap_sentences=overlap_sentences,
        )

        # metadate generation
        for piece in section_chunks:
            piece["chunk_id"] = f"{book_slug}-{chunk_global_id}"
            piece["book_title"] = book_title
            piece["book_slug"] = book_slug
            piece["chapter"] = current_chapter
            piece["section"] = section_title
            piece["section_index"] = section_index
            piece["source_path"] = master_source_path
            piece["chunk_type"] = piece.get("chunk_type", "semantic_group")
            piece["token_count"] = count_tokens(piece["text"])
            piece["equations"] = extract_equations_from_text(piece["text"])
            chunks.append(piece)
            chunk_global_id += 1

    return chunks

# =============================================================================

# -------------------- Master ingestion controller ----------------------------

#
def build_index(
    pdf_dir: str,
    persist_dir: str,
    collection_name: str,
    parser_choice: str,
    embedding_model_name: str,
    manifest_path: str,
    bm25_path: str,
    bm25_json_path: str,
    equation_index_path: str,
    rebuild: bool = True,
    similarity_threshold: float = 0.62,
    semantic_max_sentences: int = 8,
    semantic_max_tokens: int = 420,
    overlap_sentences: int = 2,
    fixed_max_tokens: int = 240,
    fixed_overlap_tokens: int = 40,
) -> Dict[str, Any]:
    
    """
    Orchestrates the end-to-end RAG ingestion pipeline.
    
    Dynamically groups and sequentially stitches partitioned text files into continuous 
    streams, executes hybrid chunking algorithms, updates a persistent ChromaDB collection, 
    generates a serialized BM25 bundle, and logs an operations manifest map.
    
    Returns:
        Dict[str, Any]: A structural tracking summary of paths, chunk metrics, and ingestion counts.
    """

    pdf_files = load_textbooks(pdf_dir)
    if not pdf_files:
        raise RuntimeError(f"No PDF files found in: {pdf_dir}")

    embedding_model = SentenceTransformer(embedding_model_name)
    all_chunks = []
    books_summary = []

    # Step 1: Group split files dynamically by their original textbook titles
    book_groups = defaultdict(list)
    # loops through scanned files for the string "_part_"
    for pdf_path in pdf_files:
        if "_part_" in pdf_path.stem:
            # strips out the part numbers and groups
            # matching paths into a dictionary list indexed by the original book's master title.
            # This ensures segmented files are recognized as segments of a single unified book entity.
            base_book_name = pdf_path.stem.split("_part_")[0].replace("_", " ").strip()
            book_groups[base_book_name].append(pdf_path)
        else:
            clean_name = pdf_path.stem.replace("_", " ").strip()
            book_groups[clean_name].append(pdf_path)

    print(f" Identified {len(book_groups)} unique books across file parts.")

    # Step 2: Loop through each unified master textbook group
    for book_title, part_paths in book_groups.items():
        print(f"\n Processing text stream pipeline for: {book_title}")
        stitched_markdown_list = []
        
        # Sort parts sequentially (part_01, part_02...) to maintain textbook order
        part_paths = sorted(part_paths, key=lambda p: p.stem.lower())
        master_source_path = str(part_paths[0].resolve())  # Reference for root tracking
        
        for part_path in part_paths:
            print(f"  Parsing layout file: {part_path.name}")
            # Parse only to generate raw text string blocks (safeguards hardware RAM limits)
            part_markdown = parse_pdf_to_markdown(str(part_path), parser_choice=parser_choice)
            stitched_markdown_list.append(part_markdown)
            
        # Step 3: Stitch text pieces together with a divider annotation
        full_book_markdown = "\n\n\n\n".join(stitched_markdown_list)
        print(f" Stitched size: {len(full_book_markdown)} chars. Healing boundaries...")
        
        # Step 4: Pass continuous unbroken book stream directly into the chunking engine
        print("Executing semantic model sequence over full continuous corpus...")

        book_chunks = build_chunks_for_text_stream(
            markdown_text=full_book_markdown,
            book_title=book_title,
            master_source_path=master_source_path,
            embedding_model=embedding_model,
            similarity_threshold=similarity_threshold,
            semantic_max_sentences=semantic_max_sentences,
            semantic_max_tokens=semantic_max_tokens,
            overlap_sentences=overlap_sentences,
            fixed_max_tokens=fixed_max_tokens,
            fixed_overlap_tokens=fixed_overlap_tokens
        )
        
        all_chunks.extend(book_chunks)
        books_summary.append({
            "book_title": book_title,
            "source_path": master_source_path,
            "chunk_count": len(book_chunks),
            "markdown_char_count": len(full_book_markdown),
        })

        print(f" Added {len(book_chunks)} continuous elements to collection.")

    if not all_chunks:
        raise RuntimeError("No chunks produced")

    ensure_dir(persist_dir)

    # embed chunks and populate semantic chroma db vector index
    upsert_into_chroma(
        chunks=all_chunks,
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedding_model_name=embedding_model_name,
        rebuild=rebuild,
    )

    # takes the massive list of all text chunks generated across all textbooks,
    # and processes them through your custom tokenization logic.
    bm25_bundle = build_bm25_bundle(all_chunks)

    # takes that heavy mathematical engine bundle sitting in your computer's RAM,
    # and freezes it to disk.
    save_bm25_bundle(bm25_bundle, bm25_path)

    # Export a cross-platform, human-readable JSON backup of the text and tokenized corpus
    save_bm25_bundle_json(bm25_bundle, bm25_json_path)

    # Scan text chunks to extract all unique mathematical formulas and map their structural metadata
    equation_index = build_equation_index(all_chunks)
    total_equations = len(equation_index["equations"])
    # Save the structured equation registry to disk as a readable, UTF-8 encoded JSON file
    save_equation_index_json(equation_index, equation_index_path)

    # Writes a high-level JSON/system tracking summary
    # mapping your source parameters, file character volumes, chunk distributions, and library paths
    save_manifest(
        manifest_path,
        pdf_dir=pdf_dir,
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedding_model_name=embedding_model_name,
        parser_choice=parser_choice,
        total_books=len(book_groups),  # Represents actual distinct textbooks
        total_chunks=len(all_chunks),
        books_summary=books_summary,
    )

    return {
        "pdf_dir": str(Path(pdf_dir).resolve()),
        "persist_dir": str(Path(persist_dir).resolve()),
        "collection_name": collection_name,
        "embedding_model_name": embedding_model_name,
        "parser_choice": parser_choice,
        "total_books": len(book_groups),
        "total_chunks": len(all_chunks),
        "total_equations": total_equations,
        "books_summary": books_summary,
        "manifest_path": str(Path(manifest_path).resolve()),
        "bm25_path": str(Path(bm25_path).resolve()),
        "equation_index_path": str(Path(equation_index_path).resolve()),
    }

# ==========================================================================

# ----------------------- CLI layer----------------------------

# reads command line prompts
def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Build persistent ChromaDB for macroeconomics textbooks.")
    parser.add_argument("--pdf-dir", default=DEFAULT_PDF_DIR)
    parser.add_argument("--persist-dir", default=DEFAULT_PERSIST_DIR)
    parser.add_argument("--collection-name", default=DEFAULT_COLLECTION)
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--bm25-path", default=DEFAULT_BM25_PATH)
    parser.add_argument("--bm25-json-path", default=DEFAULT_BM25_JSON_PATH)
    parser.add_argument("--equation-index-path", default=DEFAULT_EQUATION_INDEX_PATH)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--parser", default="mineru")
    parser.add_argument("--no-rebuild", action="store_true")
    parser.add_argument("--similarity-threshold", type=float, default=0.62)
    parser.add_argument("--semantic-max-sentences", type=int, default=15)
    parser.add_argument("--semantic-max-tokens", type=int, default=800)
    parser.add_argument("--overlap-sentences", type=int, default=3)
    parser.add_argument("--fixed-max-tokens", type=int, default=240)
    parser.add_argument("--fixed-overlap-tokens", type=int, default=40)

    return parser.parse_args()

# ===================================================================================

# --------------------- main function-------------------------------

# application entry point
def main() -> None:
    args = parse_args()
    result = build_index(
        pdf_dir=args.pdf_dir,
        persist_dir=args.persist_dir,
        collection_name=args.collection_name,
        parser_choice=args.parser,
        embedding_model_name=args.embedding_model,
        manifest_path=args.manifest_path,
        bm25_path=args.bm25_path,
        bm25_json_path=args.bm25_json_path,
        equation_index_path=args.equation_index_path,
        rebuild=not args.no_rebuild,
        similarity_threshold=args.similarity_threshold,
        semantic_max_sentences=args.semantic_max_sentences,
        semantic_max_tokens=args.semantic_max_tokens,
        overlap_sentences=args.overlap_sentences,
        fixed_max_tokens=args.fixed_max_tokens,
        fixed_overlap_tokens=args.fixed_overlap_tokens,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()