
#!/usr/bin/env python3
"""
Macro Assist: a Streamlit tutor for multiple macroeconomics textbooks with web fallback.

Retrieval flow:
1. Optimize the query
2. Search persistent ChromaDB + BM25 index
3. Rerank with BAAI/bge-reranker-base
4. Prefer textbook context
5. Use web fallback only when the query is economics-related and the books are insufficient
6. Otherwise return a safe refusal
"""

from __future__ import annotations

import json
import os
import pickle
import re
import time
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlencode, urlparse
import sympy as sp
from sympy import SympifyError

import numpy as np
import requests
import streamlit as st
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

try:
    import chromadb
except Exception:
    chromadb = None

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

try:
    from duckduckgo_search import DDGS
except Exception:
    DDGS = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

import google.generativeai as genai

# Optional parsers are kept only for debugging / future extension.
try:
    from docling.document_converter import DocumentConverter
except Exception:
    DocumentConverter = None

try:
    from llama_parse import LlamaParse
except Exception:
    LlamaParse = None


# =========================
# App setup and config
# =========================
st.set_page_config(page_title="Macro Assist", page_icon="📘", layout="wide")
st.title("📘 Macro Assist")
st.caption("Macroeconomics tutor with persistent textbook retrieval, reranking, and web fallback.")

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
LLAMA_API_KEY = os.getenv("LLAMA_API_KEY", "").strip()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PDF_DIR = os.getenv("MACRO_PDF_DIR", str(PROJECT_ROOT / "data" / "macro_textbooks"))
DEFAULT_PERSIST_DIR = os.getenv("MACRO_CHROMA_DIR", str(PROJECT_ROOT / "data" / "index" / "macro_chroma_db"))
DEFAULT_MANIFEST_PATH = os.getenv("MACRO_MANIFEST_PATH", str(PROJECT_ROOT / "data" / "index" / "macro_index_manifest.json"))
DEFAULT_BM25_PATH = os.getenv("MACRO_BM25_PATH", str(PROJECT_ROOT / "data" / "index" / "macro_bm25.pkl"))
DEFAULT_EQUATION_INDEX_PATH = os.getenv("MACRO_EQUATION_INDEX_PATH", str(PROJECT_ROOT / "data" / "index" / "macro_equation_index.json"))
DEFAULT_EMBEDDING_MODEL = os.getenv("MACRO_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5" )
DEFAULT_COLLECTION = os.getenv("MACRO_COLLECTION_NAME", "macro_textbooks")
DEFAULT_RERANKER_MODEL = os.getenv("MACRO_RERANKER_MODEL", "BAAI/bge-reranker-base")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

WEB_FALLBACK_THRESHOLD_DEFAULT = 0.25
NO_INFO_RESPONSE = "Sorry, I do not have enough information to answer that from the macroeconomics books or reliable fallback sources."


# =========================
# Text utilities
# =========================

def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\u200b", " ").replace("\ufeff", " ")
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+(?:[-'][a-zA-Z0-9]+)*", (text or "").lower())


def count_tokens(text: str) -> int:
    return max(1, len(tokenize(text)))


def _safe_label(text: Optional[str], fallback: str = "Unknown") -> str:
    text = _normalize_text(text or "")
    return text if text else fallback


def _remove_surrounding_quotes(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'", "`"}:
        return text[1:-1].strip()
    return text

# ===================================
# Helper for dealing with equations
# ===================================

_EQUATION_BLOCK_RE = re.compile(
    r"(\$\$.*?\$\$|\\\[.*?\\\])",
    flags=re.DOTALL,
)

def _render_answer_with_equations(answer: str) -> None:
    """
    Renders normal markdown normally, and renders only equation blocks with st.latex().
    This preserves general text formatting and improves equation presentation.
    """
    answer = (answer or "").strip()
    if not answer:
        return

    last_idx = 0

    for match in _EQUATION_BLOCK_RE.finditer(answer):
        before = answer[last_idx:match.start()]
        if before.strip():
            st.markdown(before)

        block = match.group(0).strip()

        # Remove delimiters:
        # $$ ... $$
        # \[ ... \]
        if block.startswith("$$") and block.endswith("$$"):
            latex = block[2:-2].strip()
        elif block.startswith("\\[") and block.endswith("\\]"):
            latex = block[2:-2].strip()
        else:
            latex = block

        if latex:
            st.latex(latex)

        last_idx = match.end()

    tail = answer[last_idx:]
    if tail.strip():
        st.markdown(tail)

def _standardize_equation_delimiters(text: str) -> str:
    text = (text or "")
    text = re.sub(
        r"\\\[(.*?)\\\]",
        r"$$\1$$",
        text,
        flags=re.DOTALL,
    )
    return text

# =========================
# Macro numerical solving
# =========================

_MACRO_NUMERIC_HINTS = {
    "calculate", "compute", "find", "determine", "solve", "evaluate",
    "what is the value", "how much", "how many", "percent", "percentage",
    "inflation", "gdp", "deflator", "multiplier", "cpi", "real gdp",
    "nominal gdp", "per capita", "growth rate", "money multiplier",
    "current account", "net exports", "net factor payments",
    "output gap", "unemployment", "interest rate", "fiscal", "monetary",
}

_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")

_VALUE_RE = re.compile(
    r"\b([A-Za-z][A-Za-z0-9_]*)\s*=\s*(-?\$?\d[\d,]*(?:\.\d+)?%?)\b"
)

_FORMULA_RE = re.compile(
    r"(?m)^\s*([A-Za-z][A-Za-z0-9_()\s\-]{0,60})\s*=\s*([^\n]{3,200})\s*$"
)

_NUMBER_TEXT_RE = r"-?\$?\d[\d,]*(?:\.\d+)?%?"

_VALUE_ALIASES: Dict[str, Sequence[str]] = {
    "C": ("consumption", "consumer spending", "household spending"),
    "I": ("investment", "planned investment"),
    "G": ("government spending", "government purchases", "govt spending"),
    "X": ("exports",),
    "M": ("imports",),
    "NX": ("net exports",),
    "NOMINAL_GDP": ("nominal gdp", "nominal output", "current dollar gdp"),
    "REAL_GDP": ("real gdp", "real output"),
    "GDP_DEFLATOR": ("gdp deflator", "deflator"),
    "CPI": ("cpi", "consumer price index"),
    "CPI_OLD": ("old cpi", "initial cpi", "previous cpi", "last year's cpi", "cpi old"),
    "CPI_NEW": ("new cpi", "final cpi", "current cpi", "this year's cpi", "cpi new"),
    "PRICE_OLD": ("old price", "initial price", "previous price"),
    "PRICE_NEW": ("new price", "final price", "current price"),
    "COST_CURRENT": ("current basket cost", "cost of current basket", "current basket"),
    "COST_BASE": ("base basket cost", "cost of base basket", "base basket"),
    "OLD": ("old value", "initial value", "previous value", "beginning value"),
    "NEW": ("new value", "final value", "current value", "ending value"),
    "RESERVE_RATIO": ("reserve ratio", "required reserve ratio", "rr"),
    "MONETARY_BASE": ("monetary base", "high powered money", "base money"),
    "MPC": ("mpc", "marginal propensity to consume"),
    "TAX_MULTIPLIER": ("tax multiplier",),
}

_NOISY_EQUATION_MARKERS = {
    "<td", "</td", "<tr", "</tr", "rowspan", "colspan", "www", "http",
    "instructor", "powerpoint", "test bank", "supplement", "copyright",
}


@dataclass(frozen=True)
class IntentSignals:
    task: str
    is_macro: bool
    is_numerical: bool
    wants_formula: bool
    wants_graph: bool
    wants_data: bool
    wants_web: bool
    reasons: Tuple[str, ...]


def is_macro_numerical_problem(query: str) -> bool:
    q = _normalize_text(query).lower()
    if not q:
        return False

    has_number = bool(re.search(r"\d", q))
    has_hint = any(hint in q for hint in _MACRO_NUMERIC_HINTS)
    has_assignment = q.count("=") >= 1

    return has_number and (has_hint or has_assignment)


def _parse_number(value: str) -> Tuple[float, bool]:
    raw = (value or "").strip()
    is_percent = raw.endswith("%")
    raw = raw.replace("$", "").replace(",", "").rstrip("%")
    return float(raw), is_percent


def _store_value(values: Dict[str, float], key: str, value: float, is_percent: bool = False) -> None:
    key = key.upper()
    if is_percent and key in {"RESERVE_RATIO", "MPC", "TAX_RATE"} and abs(value) > 1:
        value = value / 100.0
    values.setdefault(key, value)


def extract_values_from_query(query: str) -> Dict[str, float]:
    """
    Extracts assignments and common macro variable phrases like:
    C = 400
    CPI1 = 120
    reserve ratio is 10%
    nominal GDP = 500 and GDP deflator = 125
    NX = -20
    """
    values: Dict[str, float] = {}
    for var, val in _VALUE_RE.findall(query or ""):
        parsed, is_percent = _parse_number(val)
        _store_value(values, var, parsed, is_percent)

    q = _normalize_text(query)
    for canonical, aliases in _VALUE_ALIASES.items():
        for alias in aliases:
            escaped = re.escape(alias)
            patterns = [
                rf"\b{escaped}\b\s*(?:=|is|are|was|were|of|:)?\s*({_NUMBER_TEXT_RE})",
                rf"({_NUMBER_TEXT_RE})\s*(?:for|as|in)?\s*\b{escaped}\b",
            ]
            for pattern in patterns:
                match = re.search(pattern, q, flags=re.IGNORECASE)
                if match:
                    parsed, is_percent = _parse_number(match.group(1))
                    _store_value(values, canonical, parsed, is_percent)
                    break

    cpi_year_patterns = [
        rf"\bcpi\b[^\d]{{0,30}}((?:19|20)\d{{2}})[^\d-]{{0,30}}({_NUMBER_TEXT_RE})",
        rf"\b((?:19|20)\d{{2}})[^\d]{{0,30}}\bcpi\b[^\d-]{{0,30}}({_NUMBER_TEXT_RE})",
    ]
    for pattern in cpi_year_patterns:
        for match in re.finditer(pattern, q, flags=re.IGNORECASE):
            year = match.group(1)
            parsed, is_percent = _parse_number(match.group(2))
            _store_value(values, f"CPI{year}", parsed, is_percent)

    cpi_year_values = []
    for key, value in values.items():
        if re.fullmatch(r"CPI(?:19|20)\d{2}", key):
            year = int(key[-4:])
            cpi_year_values.append((year, value))
    if len(cpi_year_values) >= 2:
        cpi_year_values.sort()
        values.setdefault("CPI_OLD", cpi_year_values[0][1])
        values.setdefault("CPI_NEW", cpi_year_values[-1][1])

    return values


def _normalize_formula_key(formula_text: str) -> str:
    text = _normalize_text(formula_text).lower()
    text = text.replace("$$", "").replace("\\[", "").replace("\\]", "")
    text = text.replace("\\(", "").replace("\\)", "")
    text = re.sub(r"\\left|\\right", "", text)
    text = re.sub(r"\s*([=+\-*/^()])\s*", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _is_plausible_formula(formula_text: str) -> bool:
    text = _normalize_text(formula_text)
    lowered = text.lower()
    if "=" not in text or len(text) < 5 or len(text) > 240:
        return False
    if any(marker in lowered for marker in _NOISY_EQUATION_MARKERS):
        return False
    lhs, rhs = text.split("=", 1)
    if not re.search(r"[A-Za-z]", lhs) or not re.search(r"[A-Za-z0-9]", rhs):
        return False
    if len(re.findall(r"[A-Za-z]{4,}", rhs)) > 8 and not re.search(r"[+\-*/^()\\]", rhs):
        return False
    return True


def _equation_source_from_item(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "book_title": item.get("book_title", ""),
        "book_slug": item.get("book_slug", ""),
        "chapter": item.get("chapter", ""),
        "section": item.get("section", ""),
        "section_index": item.get("section_index", 0),
        "source_path": item.get("source_path", ""),
        "chunk_type": item.get("chunk_type", ""),
        "chunk_id": item.get("chunk_id"),
        "source": "equation_index",
    }


def build_grouped_equation_index(
    equation_index_path: str,
    dense_model: SentenceTransformer,
) -> Dict[str, Any]:
    path = Path(equation_index_path)
    if not path.exists():
        return {"formulas": [], "embeddings": np.empty((0, 0), dtype=np.float32)}

    with open(path, "r", encoding="utf-8") as f:
        raw_index = json.load(f)

    grouped: Dict[str, Dict[str, Any]] = {}
    for item in raw_index.get("equations", []):
        formula = _normalize_text(item.get("equation", ""))
        if not _is_plausible_formula(formula):
            continue

        key = _normalize_formula_key(formula)
        group = grouped.setdefault(
            key,
            {
                "formula": formula,
                "normalized_formula": key,
                "sources": [],
                "source_keys": set(),
            },
        )

        source = _equation_source_from_item(item)
        source_key = (
            source.get("book_title", ""),
            source.get("chapter", ""),
            source.get("section", ""),
            source.get("chunk_id", ""),
        )
        if source_key not in group["source_keys"]:
            group["sources"].append(source)
            group["source_keys"].add(source_key)

    formulas = []
    for group in grouped.values():
        group.pop("source_keys", None)
        formulas.append(group)

    formulas.sort(key=lambda item: (item["normalized_formula"], item["formula"]))
    if not formulas:
        return {"formulas": [], "embeddings": np.empty((0, 0), dtype=np.float32)}

    embedding_texts = [item["formula"] for item in formulas]
    embeddings = dense_model.encode(
        embedding_texts,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return {"formulas": formulas, "embeddings": np.asarray(embeddings, dtype=np.float32)}


def search_equations(
    query: str,
    equation_index: Dict[str, Any],
    dense_model: SentenceTransformer,
    top_k: int = 6,
    min_similarity: float = 0.32,
) -> List[Dict[str, Any]]:
    formulas = equation_index.get("formulas", []) if equation_index else []
    embeddings = equation_index.get("embeddings") if equation_index else None
    if not formulas or embeddings is None or len(embeddings) == 0:
        return []

    query_embedding = dense_model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
    query_embedding = np.asarray(query_embedding, dtype=np.float32)
    scores = np.dot(embeddings, query_embedding)
    top_indices = np.argsort(scores)[::-1][:top_k]

    matches: List[Dict[str, Any]] = []
    for idx in top_indices:
        score = float(scores[idx])
        if score < min_similarity:
            continue
        item = dict(formulas[int(idx)])
        item["equation_similarity"] = score
        matches.append(item)
    return matches


def extract_candidate_formulas(chunks: List[Dict[str, Any]]) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Pull likely formulas from retrieved textbook chunks.
    Returns [(formula_text, source_chunk), ...]
    """
    formulas: List[Tuple[str, Dict[str, Any]]] = []
    seen = set()

    for chunk in chunks:
        text = chunk.get("text", "") or ""
        for lhs, rhs in _FORMULA_RE.findall(text):
            lhs_clean = _normalize_text(lhs)
            rhs_clean = _normalize_text(rhs)

            # Keep only plausible equation-like lines.
            if not lhs_clean or not rhs_clean:
                continue
            if len(rhs_clean) < 3:
                continue

            formula = f"{lhs_clean} = {rhs_clean}"
            key = formula.lower()
            if key in seen:
                continue

            seen.add(key)
            formulas.append((formula, chunk))

    return formulas


def _sympy_safe_locals(values: Dict[str, float], formula_text: str = "") -> Dict[str, Any]:
    local_dict: Dict[str, Any] = {}
    names = set(values.keys())
    names.update(re.findall(r"\b[A-Za-z][A-Za-z0-9_]*\b", formula_text or ""))
    for name in names:
        sym = sp.Symbol(name)
        local_dict[name] = sym
        local_dict[name.lower()] = sym
        local_dict[name.upper()] = sym
    return local_dict


def _clean_formula_for_sympy(formula_text: str) -> str:
    text = _normalize_text(formula_text)
    text = text.replace("−", "-").replace("×", "*").replace("÷", "/")
    text = text.replace("^", "**")
    text = re.sub(r"\\left|\\right", "", text)
    text = text.replace("\\times", "*").replace("\\cdot", "*").replace("\\div", "/")
    text = text.replace("\\(", "").replace("\\)", "").replace("\\[", "").replace("\\]", "").replace("$$", "")
    return text


def solve_formula_text(formula_text: str, values: Dict[str, float]) -> Optional[float]:
    """
    Solves one missing symbol in an equation after substituting known values.
    """
    try:
        if "=" not in formula_text:
            return None

        formula_text = _clean_formula_for_sympy(formula_text)
        lhs_text, rhs_text = formula_text.split("=", 1)
        if not re.search(r"[a-zA-Z]", lhs_text + rhs_text):
            return None

        local_dict = _sympy_safe_locals(values, formula_text)
        lhs_expr = sp.sympify(lhs_text.strip(), locals=local_dict)
        rhs_expr = sp.sympify(rhs_text.strip(), locals=local_dict)

        subs = {}
        for name, value in values.items():
            for candidate in (name, name.upper(), name.lower()):
                symbol = local_dict.get(candidate)
                if symbol is not None:
                    subs[symbol] = value

        free_symbols = (lhs_expr.free_symbols | rhs_expr.free_symbols) - set(subs.keys())
        if len(free_symbols) != 1:
            return None

        target = next(iter(free_symbols))
        solved = sp.solve(sp.Eq(lhs_expr, rhs_expr), target)
        if not solved:
            return None
        result = solved[0].subs(subs)
        if getattr(result, "free_symbols", set()):
            return None
        return float(sp.N(result))

    except (SympifyError, ValueError, ZeroDivisionError, TypeError):
        return None
    except Exception:
        return None


def _solver_payload(
    *,
    formula: str,
    result: float,
    values: Dict[str, float],
    method: str,
    target: str = "",
    source_chunks: Optional[List[Dict[str, Any]]] = None,
    equation_matches: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    return {
        "formula": formula,
        "result": result,
        "values": values,
        "method": method,
        "target": target,
        "source_chunks": source_chunks or [],
        "equation_matches": equation_matches or [],
    }


def solve_hardcoded_macro_formula(raw_query: str, values: Dict[str, float]) -> Optional[Dict[str, Any]]:
    q = _normalize_text(raw_query).lower()

    if all(k in values for k in ("C", "I", "G")) and ("NX" in values or all(k in values for k in ("X", "M"))):
        nx = values["NX"] if "NX" in values else values["X"] - values["M"]
        result = values["C"] + values["I"] + values["G"] + nx
        formula = "GDP = C + I + G + NX"
        if "NX" not in values:
            formula = "GDP = C + I + G + (X - M)"
        return _solver_payload(formula=formula, result=result, values=values, method="hardcoded", target="GDP")

    if "real gdp" in q and all(k in values for k in ("NOMINAL_GDP", "GDP_DEFLATOR")):
        result = values["NOMINAL_GDP"] / values["GDP_DEFLATOR"] * 100.0
        return _solver_payload(
            formula="Real GDP = (Nominal GDP / GDP Deflator) * 100",
            result=result,
            values=values,
            method="hardcoded",
            target="REAL_GDP",
        )

    if ("cpi" in q or "consumer price index" in q) and all(k in values for k in ("COST_CURRENT", "COST_BASE")):
        result = values["COST_CURRENT"] / values["COST_BASE"] * 100.0
        return _solver_payload(
            formula="CPI = (Cost of basket in current period / Cost of basket in base period) * 100",
            result=result,
            values=values,
            method="hardcoded",
            target="CPI",
        )

    if ("inflation" in q or "price level" in q) and all(k in values for k in ("CPI_NEW", "CPI_OLD")):
        result = (values["CPI_NEW"] - values["CPI_OLD"]) / values["CPI_OLD"] * 100.0
        return _solver_payload(
            formula="Inflation rate = ((CPI_new - CPI_old) / CPI_old) * 100",
            result=result,
            values=values,
            method="hardcoded",
            target="INFLATION_RATE",
        )

    if ("growth" in q or "percentage change" in q or "percent change" in q) and all(k in values for k in ("NEW", "OLD")):
        result = (values["NEW"] - values["OLD"]) / values["OLD"] * 100.0
        return _solver_payload(
            formula="Growth rate = ((New value - Old value) / Old value) * 100",
            result=result,
            values=values,
            method="hardcoded",
            target="GROWTH_RATE",
        )

    if "money multiplier" in q and "RESERVE_RATIO" in values:
        result = 1.0 / values["RESERVE_RATIO"]
        return _solver_payload(
            formula="Money multiplier = 1 / Reserve ratio",
            result=result,
            values=values,
            method="hardcoded",
            target="MONEY_MULTIPLIER",
        )

    if "money supply" in q and all(k in values for k in ("MONETARY_BASE", "RESERVE_RATIO")):
        result = values["MONETARY_BASE"] / values["RESERVE_RATIO"]
        return _solver_payload(
            formula="Money supply = Monetary base * (1 / Reserve ratio)",
            result=result,
            values=values,
            method="hardcoded",
            target="MONEY_SUPPLY",
        )

    if "multiplier" in q and "MPC" in values:
        result = 1.0 / (1.0 - values["MPC"])
        return _solver_payload(
            formula="Government spending multiplier = 1 / (1 - MPC)",
            result=result,
            values=values,
            method="hardcoded",
            target="FISCAL_MULTIPLIER",
        )

    return None


def solve_macro_numerical_problem(
    raw_query: str,
    reranked_chunks: List[Dict[str, Any]],
    equation_index: Dict[str, Any],
    dense_model: SentenceTransformer,
) -> Optional[Dict[str, Any]]:
    values = extract_values_from_query(raw_query)
    if not values:
        return None

    equation_matches = search_equations(raw_query, equation_index, dense_model, top_k=8)

    hardcoded = solve_hardcoded_macro_formula(raw_query, values)
    if hardcoded is not None:
        hardcoded["equation_matches"] = equation_matches
        if equation_matches:
            hardcoded["source_chunks"] = equation_matches[0].get("sources", [])
        return hardcoded

    candidate_formulas: List[Tuple[str, Dict[str, Any]]] = []
    for match in equation_matches:
        representative = {
            "book_title": " / ".join(
                dict.fromkeys(_safe_label(src.get("book_title"), "") for src in match.get("sources", []) if src.get("book_title"))
            ),
            "section": "Grouped Equation Index",
            "chapter": "",
            "chunk_id": match.get("normalized_formula"),
            "source": "equation_index",
            "equation_similarity": match.get("equation_similarity"),
            "grouped_sources": match.get("sources", []),
        }
        candidate_formulas.append((match["formula"], representative))

    candidate_formulas.extend(extract_candidate_formulas(reranked_chunks))

    for formula_text, source_chunk in candidate_formulas:
        result = solve_formula_text(formula_text, values)
        if result is not None:
            grouped_sources = source_chunk.get("grouped_sources") or [source_chunk]
            return {
                "formula": formula_text,
                "result": result,
                "values": values,
                "method": "equation_index" if source_chunk.get("source") == "equation_index" else "retrieved_chunk_formula",
                "source_chunks": grouped_sources,
                "equation_matches": equation_matches,
            }

    return None

# =========================
# LLM provider helpers
# =========================

def _provider_api_key(provider: str) -> str:
    provider_norm = _normalize_text(provider).lower()
    if provider_norm == "openai":
        return OPENAI_API_KEY
    if provider_norm == "groq":
        return GROQ_API_KEY
    if provider_norm == "deepseek":
        return DEEPSEEK_API_KEY
    return GEMINI_API_KEY


def _call_text_llm(
    provider: str,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    top_p: float = 0.95,
    max_output_tokens: int = 256,
) -> str:
    
    provider_norm = _normalize_text(provider).lower()
    model_name = _normalize_text(model_name)

    if provider_norm == "gemini":
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is missing.")
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_prompt,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                top_p=top_p,
                max_output_tokens=max_output_tokens,
            ),
        )
        response = model.generate_content(user_prompt)
        return (getattr(response, "text", "") or "")

    if OpenAI is None:
        raise RuntimeError("openai package is not installed.")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    if provider_norm == "openai":
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is missing.")
        
        client = OpenAI(api_key=OPENAI_API_KEY)

        request_kwargs = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        }

        if model_name.startswith("gpt-5"):
            request_kwargs["max_completion_tokens"] = max_output_tokens
        else:
            request_kwargs["max_tokens"] = max_output_tokens

        response = client.chat.completions.create(**request_kwargs)
        return (getattr(response.choices[0].message, "content", "") or "")
    
    if provider_norm == "deepseek":
        if not DEEPSEEK_API_KEY:
            raise RuntimeError("DEEPSEEK_API_KEY is missing.")
        client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com",
        )
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_output_tokens,
        )
        return (getattr(response.choices[0].message, "content", "") or "")

    if provider_norm == "groq":
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is missing.")
        client = OpenAI(
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
        )
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_output_tokens,
        )
        return (getattr(response.choices[0].message, "content", "") or "")

    raise ValueError(f"Unsupported provider: {provider}")


# =========================
# Prompts
# =========================

OPTIMIZER_SYSTEM_PROMPT = """
You are a query optimizer for a Macroeconomics tutor.

Task:
1. Identify the economic concept, model, graph, theorem, or policy question.
2. Remove filler, greetings, and conversational wording.
3. Expand with standard macroeconomics terminology and synonyms.
4. Preserve named models, variables, authors, chapter concepts, and textbook-style wording.
5. Do not answer the question.
6. Output only one compact search query.

Rules:
- Prefer canonical macro terms.
- Keep the output short, dense, and search-friendly.
- If the user query is already strong, preserve it with minimal cleanup.
"""

GENERATION_SYSTEM_PROMPT = """
You are Macro Assist, an expert Macroeconomics tutor.

========================================
KNOWLEDGE PRIORITY
========================================

Use information in this order:

1. Textbook context
2. Web fallback context (only if textbook context is insufficient)
3. No other knowledge

Never use knowledge that is not supported by the provided context.

========================================
GROUNDING RULES
========================================

- Use textbook context whenever available.
- If textbook context partially answers the question, answer using the available information and clearly identify what is missing.
- Use web fallback only when textbook evidence is insufficient and the question remains economics-related.
- If web fallback is used, explicitly state:
  "According to external economics sources..."

- Never invent:
  - facts
  - equations
  - definitions
  - chapter names
  - section names
  - citations
  - economic data

- If evidence is insufficient, reply exactly:

Sorry, I do not have enough information to answer that from the macroeconomics books or reliable fallback sources.

========================================
TEACHING STYLE
========================================

Explain concepts as a university economics tutor.

When answering:

- Start with a direct answer.
- Explain the economic intuition.
- Use simple language.
- Break complex concepts into steps.
- Avoid unnecessary jargon.
- Never dump textbook passages verbatim.
- Rewrite concepts in a student-friendly way.

========================================
FORMATTING RULES
========================================

Always use Markdown.

- For equations, use display math with proper LaTeX delimiters.
  Example:
  $$
  GDP = C + I + G + NX
  $$

- Do not put equations inside normal paragraphs when a display equation is better.
- Keep normal explanatory text in markdown.
Formatting Guidelines

- Use markdown.
- Use headings only when they improve readability.
- Use bullet points where helpful.
- Use tables for comparisons.
- Use formulas in separate blocks.
- Keep answers concise unless the user asks for detail.
- Avoid repeating section headings unnecessarily.

At the very end mention the sources,

## Sources
List the textbook titles and sections used.
If web fallback was used, clearly indicate:
- External Economics Source

========================================
SPECIAL CASES
========================================

For comparison questions:
- Use tables.

For graph-based questions:
- Explain:
  - X-axis
  - Y-axis
  - Curve shifts
  - Movements along curves

For policy questions:
- Separate:
  - Short-run effects
  - Long-run effects

For formula questions:
- Show:
  - Formula
  - Variable definitions
  - Interpretation

For calculation questions:
- Show every step.
- Show the final answer clearly.

For numerical problems:

- Prioritize calculations.
- Keep explanations concise.
- Do not generate lengthy conceptual discussions.
- Show formulas and substitutions.
- Do not place LaTeX equations inside markdown tables.
- Show final answers.
- If a computed result is provided in the context, use that result.
- Do not recompute arithmetic.
- Briefly explain the formula used.
- Briefly explain substitutions.
- Briefly explain the economic meaning of the result.
-If using tables:
    - Show only final values in tables.
    - Show formulas and calculations outside the table.
"""

OPTIMIZER_CONFIG = genai.GenerationConfig(temperature=0.0, top_p=0.8, max_output_tokens=64)
GENERATION_CONFIG = genai.GenerationConfig(temperature=0.2, top_p=0.95, max_output_tokens=800)

LLM_PROVIDER_OPTIONS = ["Gemini", "OpenAI", "Groq", "DeepSeek"]


# =========================
# File paths and index loading
# =========================

def get_paths() -> Dict[str, str]:
    persist_dir = st.sidebar.text_input("ChromaDB directory", value=DEFAULT_PERSIST_DIR)
    collection_name = st.sidebar.text_input("Collection name", value=DEFAULT_COLLECTION)
    manifest_path = st.sidebar.text_input("Manifest JSON", value=DEFAULT_MANIFEST_PATH)
    bm25_path = st.sidebar.text_input("BM25 bundle", value=DEFAULT_BM25_PATH)
    equation_index_path = st.sidebar.text_input("Equation index JSON", value=DEFAULT_EQUATION_INDEX_PATH)
    embedding_model_name = st.sidebar.text_input("Embedding model", value=DEFAULT_EMBEDDING_MODEL)
    reranker_model_name = st.sidebar.text_input("Reranker model", value=DEFAULT_RERANKER_MODEL)
    return {
        "persist_dir": persist_dir,
        "collection_name": collection_name,
        "manifest_path": manifest_path,
        "bm25_path": bm25_path,
        "equation_index_path": equation_index_path,
        "embedding_model_name": embedding_model_name,
        "reranker_model_name": reranker_model_name,
    }


def read_manifest(manifest_path: str) -> Dict[str, Any]:
    path = Path(manifest_path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_manifest_cached(manifest_path: str) -> Dict[str, Any]:
    return read_manifest(manifest_path)


@st.cache_data(show_spinner=False)
def load_bm25_bundle_cached(bm25_path: str) -> Dict[str, Any]:
    path = Path(bm25_path)
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner="Loading models and equation index...")
def load_models(embedding_model_name: str, reranker_model_name: str, equation_index_path: str):
    dense_model = SentenceTransformer(embedding_model_name)
    reranker = CrossEncoder(reranker_model_name)
    equation_index = build_grouped_equation_index(equation_index_path, dense_model)
    return dense_model, reranker, equation_index


def _collection_exists(persist_dir: str, collection_name: str) -> bool:
    if chromadb is None:
        return False
    client = chromadb.PersistentClient(path=str(Path(persist_dir).resolve()))
    try:
        client.get_collection(collection_name)
        return True
    except Exception:
        return False


@st.cache_resource(show_spinner=False)
def load_collection_cached(persist_dir: str, collection_name: str):
    if chromadb is None:
        raise RuntimeError("chromadb is not installed.")
    client = chromadb.PersistentClient(path=str(Path(persist_dir).resolve()))
    return client.get_collection(collection_name)


# =========================
# Query cleaning and optimization
# =========================

_FILLER_PATTERNS = [
    r"\bhey\b",
    r"\bhi\b",
    r"\bhello\b",
    r"\bplease\b",
    r"\bcan you\b",
    r"\bcould you\b",
    r"\bhelp me\b",
    r"\btell me\b",
    r"\bi want to know\b",
    r"\bdo you know\b",
    r"\bwhat happens if\b",
    r"\bwhat if\b",
]


def _remove_filler_text(query: str) -> str:
    q = _normalize_text(query)
    for pat in _FILLER_PATTERNS:
        q = re.sub(pat, " ", q, flags=re.IGNORECASE)
    q = re.sub(r"\s+", " ", q).strip(" ?!.,;:")
    return q


def _clean_model_output(text: str) -> str:
    text = _normalize_text(text)
    text = _remove_surrounding_quotes(text)
    text = re.sub(r"^\s*(optimized\s+query|query|search\s+query|output)\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```(?:json|text)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text).strip()
    text = re.sub(r"\s*\n+\s*", " ", text).strip()
    text = re.split(r"(?<=[.!?])\s+", text)[0].strip()
    return _normalize_text(text)


def _validate_query(query: str, fallback: str) -> str:
    query = _normalize_text(query)
    if not query:
        return fallback
    words = query.split()
    if len(words) > 28:
        query = " ".join(words[:28]).strip()
    query = re.sub(r"[^\w\s\-/&,().:+%#]", " ", query)
    query = re.sub(r"\s+", " ", query).strip()
    return query if len(query) >= 3 else fallback


def _build_optimizer_prompt(raw_query: str) -> str:
    return f"""
User query:
{raw_query}

Return one dense search query optimized for macroeconomics textbook retrieval.
Include relevant model names, synonyms, and macro terminology.
No explanation, no bullets, no labels, no markdown.
""".strip()


def optimize_and_expand_query(
    raw_query: str,
    provider: str = "Gemini",
    model_name: str = "gemini-2.5-flash",
    max_retries: int = 3,
    retry_backoff: float = 1.5,
    return_metadata: bool = False,
) -> str | Tuple[str, Dict[str, Any]]:
    raw_query = _normalize_text(raw_query)
    if not raw_query:
        result = ""
        meta = {"status": "empty_input", "raw_query": ""}
        return (result, meta) if return_metadata else result

    cleaned_input = _remove_filler_text(raw_query)
    prompt = _build_optimizer_prompt(cleaned_input)

    if not _provider_api_key(provider):
        fallback = _validate_query(cleaned_input or raw_query, raw_query)
        meta = {
            "status": "fallback_no_api_key",
            "provider": provider,
            "model_name": model_name,
            "raw_query": raw_query,
            "cleaned_input": cleaned_input,
            "optimized_query": fallback,
        }
        return (fallback, meta) if return_metadata else fallback

    last_error: Optional[str] = None

    for attempt in range(1, max_retries + 1):
        try:
            candidate = _call_text_llm(
                provider=provider,
                model_name=model_name,
                system_prompt=OPTIMIZER_SYSTEM_PROMPT,
                user_prompt=prompt,
                temperature=0.0,
                top_p=0.8,
                max_output_tokens=64,
            )
            candidate = _clean_model_output(candidate)
            candidate = _validate_query(candidate, cleaned_input or raw_query)
            if not candidate:
                candidate = cleaned_input or raw_query

            meta = {
                "status": "ok",
                "provider": provider,
                "model_name": model_name,
                "raw_query": raw_query,
                "cleaned_input": cleaned_input,
                "optimized_query": candidate,
            }
            return (candidate, meta) if return_metadata else candidate
        except Exception as exc:
            last_error = str(exc)
            if attempt < max_retries:
                time.sleep(retry_backoff ** (attempt - 1))

    fallback = _validate_query(cleaned_input or raw_query, raw_query)
    meta = {
        "status": "fallback",
        "provider": provider,
        "model_name": model_name,
        "raw_query": raw_query,
        "cleaned_input": cleaned_input,
        "optimized_query": fallback,
        "error": last_error or "unknown_error",
    }
    return (fallback, meta) if return_metadata else fallback


def build_retrieval_query(raw_query: str, optimized_query: str) -> str:
    raw_query = _normalize_text(raw_query)
    optimized_query = _normalize_text(optimized_query)
    if not raw_query:
        return optimized_query
    if not optimized_query:
        return raw_query
    return f"{raw_query} {optimized_query}".strip()


# =========================
# Retrieval helpers
# =========================

def _chunk_from_metadata(chunk_id: str, document: str, metadata: Dict[str, Any], source: str = "textbook") -> Dict[str, Any]:
    return {
        "chunk_id": str(chunk_id),
        "text": document or "",
        "book_title": metadata.get("book_title", ""),
        "book_slug": metadata.get("book_slug", ""),
        "chapter": metadata.get("chapter", ""),
        "section": metadata.get("section", ""),
        "section_index": metadata.get("section_index", 0),
        "source_path": metadata.get("source_path", ""),
        "chunk_type": metadata.get("chunk_type", ""),
        "token_count": metadata.get("token_count", 0),
        "source": source,
    }


def vector_retrieve(
    query: str,
    collection,
    dense_model: SentenceTransformer,
    dense_top_k: int = 12,
) -> List[Dict[str, Any]]:
    query_embedding = dense_model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
    query_embedding = np.asarray(query_embedding, dtype=np.float32).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=dense_top_k,
        include=["documents", "metadatas", "distances"],
    )

    items: List[Dict[str, Any]] = []
    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for rank, (cid, doc, meta, dist) in enumerate(zip(ids, docs, metas, distances), start=1):
        item = _chunk_from_metadata(cid, doc or "", meta or {})
        item["dense_rank"] = rank
        item["chroma_distance"] = float(dist) if dist is not None else None
        items.append(item)

    return items


def bm25_retrieve(query: str, bm25_bundle: Dict[str, Any], sparse_top_k: int = 12) -> List[Dict[str, Any]]:
    if not bm25_bundle:
        return []

    bm25 = bm25_bundle.get("bm25")
    chunks = bm25_bundle.get("chunks", [])
    if bm25 is None or not chunks:
        return []

    scores = bm25.get_scores(tokenize(query))
    top_indices = np.argsort(scores)[::-1][:sparse_top_k]

    results: List[Dict[str, Any]] = []
    for rank, idx in enumerate(top_indices, start=1):
        if idx < 0 or idx >= len(chunks):
            continue
        chunk = dict(chunks[idx])
        chunk["sparse_rank"] = rank
        chunk["bm25_score"] = float(scores[idx])
        chunk["source"] = "textbook"
        results.append(chunk)

    return results


def hybrid_fuse_candidates(
    dense_candidates: List[Dict[str, Any]],
    sparse_candidates: List[Dict[str, Any]],
    final_top_k: int = 20,
) -> List[Dict[str, Any]]:
    rrf_scores: Dict[str, float] = {}
    candidate_map: Dict[str, Dict[str, Any]] = {}

    k = 60.0

    for rank, chunk in enumerate(dense_candidates, start=1):
        cid = str(chunk["chunk_id"])
        candidate_map[cid] = chunk
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)

    for rank, chunk in enumerate(sparse_candidates, start=1):
        cid = str(chunk["chunk_id"])
        candidate_map[cid] = {**candidate_map.get(cid, {}), **chunk}
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)

    ranked_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    fused: List[Dict[str, Any]] = []
    for cid, score in ranked_ids[:final_top_k]:
        chunk = dict(candidate_map[cid])
        chunk["rrf_score"] = float(score)
        fused.append(chunk)

    return fused


def rerank_chunks(
    query: str,
    chunks: List[Dict[str, Any]],
    reranker_model: CrossEncoder,
    top_k: int = 8,
    batch_size: int = 16,
) -> List[Dict[str, Any]]:
    if not chunks:
        return []

    sentence_pairs = [(query, chunk["text"]) for chunk in chunks]
    scores = reranker_model.predict(sentence_pairs, batch_size=batch_size, show_progress_bar=False)

    reranked: List[Dict[str, Any]] = []
    for chunk, score in zip(chunks, scores):
        enriched = dict(chunk)
        enriched["cross_encoder_score"] = float(score)
        reranked.append(enriched)

    reranked.sort(key=lambda x: x["cross_encoder_score"], reverse=True)
    return reranked[: min(top_k, len(reranked))]


def final_pipeline_retrieval(
    query: str,
    collection,
    bm25_bundle: Dict[str, Any],
    dense_model: SentenceTransformer,
    reranker_model: CrossEncoder,
    dense_top_k: int = 12,
    sparse_top_k: int = 12,
    fused_top_k: int = 10,
    rerank_top_k: int = 8,
) -> List[Dict[str, Any]]:
    dense_candidates = vector_retrieve(query, collection, dense_model, dense_top_k=dense_top_k)
    sparse_candidates = bm25_retrieve(query, bm25_bundle, sparse_top_k=sparse_top_k)
    fused = hybrid_fuse_candidates(dense_candidates, sparse_candidates, final_top_k=fused_top_k)
    return rerank_chunks(query, fused, reranker_model, top_k=rerank_top_k)


# =========================
# Economics relevance gate
# =========================

ECON_KEYWORDS = {
    "economics", "economy", "macro", "macroeconomics", "inflation", "deflation", "gdp", "g.n.p",
    "unemployment", "output gap", "recession", "depression", "fiscal", "monetary", "interest rate",
    "money supply", "aggregate demand", "aggregate supply", "ad-as", "is-lm", "phillips curve",
    "solow", "growth", "capital", "investment", "consumption", "saving", "exchange rate",
    "balance of payments", "trade deficit", "current account", "open market operation",
    "central bank", "liquidity", "currency", "exchange", "devaluation", "appreciation",
    "crowding out", "multiplier", "velocity", "nominal", "real", "labor market", "wage",
    "price level", "business cycle", "stagflation", "demand shock", "supply shock",
    "federal reserve", "interest", "opportunity cost", "aggregate expenditure",
}

ECON_MODEL_HINTS = {
    "is-lm", "ad-as", "phillips", "solow", "keynes", "fiscal policy", "monetary policy",
    "money demand", "money supply", "solow model", "sacrifice ratio", "natural rate",
}


def is_economics_related(query: str, reranked_chunks: Sequence[Dict[str, Any]] | None = None) -> bool:
    q = _normalize_text(query).lower()
    if not q:
        return False

    if any(keyword in q for keyword in ECON_KEYWORDS):
        return True
    if any(hint in q for hint in ECON_MODEL_HINTS):
        return True

    # If retrieval produced clearly relevant textbook hits, treat as economics-related.
    if reranked_chunks:
        top_score = reranked_chunks[0].get("cross_encoder_score", float("-inf"))
        if top_score is not None and top_score > 0.05:
            return True

    return False


def classify_intent(query: str, reranked_chunks: Sequence[Dict[str, Any]] | None = None) -> IntentSignals:
    q = _normalize_text(query).lower()
    reasons: List[str] = []

    is_numerical = is_macro_numerical_problem(query)
    wants_formula = any(word in q for word in ["formula", "equation", "derive", "calculate", "compute", "solve"])
    wants_graph = any(word in q for word in ["graph", "diagram", "curve", "shift", "draw", "ad-as", "is-lm", "phillips"])
    wants_data = any(word in q for word in ["latest", "current", "today", "recent", "data", "rate", "forecast", "statistics"])
    wants_web = wants_data or any(word in q for word in ["world bank", "imf", "fred", "pakistan", "sbp", "pbs"])
    is_macro = is_economics_related(query, reranked_chunks)

    if is_numerical:
        reasons.append("numeric macro calculation")
        task = "numerical"
    elif wants_graph:
        reasons.append("graph/model language")
        task = "graph_model"
    elif wants_data:
        reasons.append("data or policy/current source language")
        task = "data_policy"
    elif wants_formula:
        reasons.append("formula/equation language")
        task = "formula"
    else:
        task = "conceptual"

    if is_macro:
        reasons.append("macro keyword or confident textbook retrieval")

    return IntentSignals(
        task=task,
        is_macro=is_macro,
        is_numerical=is_numerical,
        wants_formula=wants_formula,
        wants_graph=wants_graph,
        wants_data=wants_data,
        wants_web=wants_web,
        reasons=tuple(reasons),
    )


TASK_PROMPT_OVERLAYS = {
    "conceptual": """
Task mode: conceptual tutoring.
Give a direct answer, explain intuition, and cite the textbook sections used.
""",
    "numerical": """
Task mode: step-by-step numerical solving.
Use the deterministic result if supplied. Show the formula, substitution, arithmetic result, and brief economic interpretation. Do not invent missing values.
""",
    "formula": """
Task mode: formula retrieval and explanation.
Prioritize the equation index and textbook context. Define variables, state when formulas are equivalent, and cite grouped textbook sources.
""",
    "graph_model": """
Task mode: graph/model explanation.
Explain axes, curves, movements along curves, shifts, equilibrium changes, and the economic intuition behind the comparative statics.
""",
    "data_policy": """
Task mode: data-backed policy answer.
Use textbook context for theory and web fallback only for current data. Separate short-run and long-run effects where relevant.
""",
    "web_fallback": """
Task mode: web fallback.
Use only trusted external economics sources provided in context. Say when evidence is insufficient.
""",
}


def _system_prompt_for_intent(intent_signals: Optional[IntentSignals], used_web: bool = False) -> str:
    task = "web_fallback" if used_web else (intent_signals.task if intent_signals else "conceptual")
    overlay = TASK_PROMPT_OVERLAYS.get(task, TASK_PROMPT_OVERLAYS["conceptual"])
    return GENERATION_SYSTEM_PROMPT + "\n\n" + overlay.strip()


def _equation_matches_to_chunks(matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    for rank, match in enumerate(matches, start=1):
        sources = match.get("sources", [])
        source_labels = []
        for src in sources:
            label = _extract_citation_label(src)
            if label not in source_labels:
                source_labels.append(label)
        text = (
            f"Formula: {match.get('formula', '')}\n"
            f"Found in: {'; '.join(source_labels[:8]) if source_labels else 'Equation index'}"
        )
        first_source = sources[0] if sources else {}
        chunks.append({
            "chunk_id": f"equation:{match.get('normalized_formula', rank)}",
            "text": text,
            "book_title": " / ".join(dict.fromkeys(src.get("book_title", "") for src in sources if src.get("book_title"))),
            "book_slug": first_source.get("book_slug", ""),
            "chapter": first_source.get("chapter", ""),
            "section": "Grouped Equation Index",
            "section_index": first_source.get("section_index", 0),
            "source_path": first_source.get("source_path", ""),
            "chunk_type": "grouped_equation",
            "token_count": count_tokens(text),
            "source": "equation_index",
            "equation_similarity": match.get("equation_similarity"),
            "cross_encoder_score": 10.0 - (rank * 0.01),
            "grouped_sources": sources,
        })
    return chunks


def _chunk_equation_count(chunk: Dict[str, Any]) -> int:
    if "equation_count" in chunk:
        try:
            return int(chunk.get("equation_count") or 0)
        except Exception:
            return 0
    equations = chunk.get("equations")
    return len(equations) if isinstance(equations, list) else 0


def expand_with_neighbor_chunks(
    selected_chunks: List[Dict[str, Any]],
    bm25_bundle: Dict[str, Any],
    intent_signals: Optional[IntentSignals],
    max_neighbors_per_chunk: int = 2,
) -> List[Dict[str, Any]]:
    chunks = bm25_bundle.get("chunks", []) if bm25_bundle else []
    if not selected_chunks or not chunks:
        return selected_chunks

    by_location: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for chunk in chunks:
        key = (_safe_label(chunk.get("book_slug"), "").lower(), _safe_label(chunk.get("chapter"), "").lower())
        by_location.setdefault(key, []).append(chunk)

    expanded: List[Dict[str, Any]] = []
    seen_ids = set()
    math_priority = bool(intent_signals and (intent_signals.is_numerical or intent_signals.wants_formula))

    for chunk in selected_chunks:
        cid = str(chunk.get("chunk_id", ""))
        if cid and cid not in seen_ids:
            expanded.append(chunk)
            seen_ids.add(cid)

        if chunk.get("source") == "equation_index":
            continue

        try:
            section_index = int(chunk.get("section_index") or 0)
        except Exception:
            section_index = 0
        if section_index <= 0:
            continue

        key = (_safe_label(chunk.get("book_slug"), "").lower(), _safe_label(chunk.get("chapter"), "").lower())
        neighbors = []
        for candidate in by_location.get(key, []):
            candidate_id = str(candidate.get("chunk_id", ""))
            if not candidate_id or candidate_id in seen_ids:
                continue
            try:
                delta = abs(int(candidate.get("section_index") or 0) - section_index)
            except Exception:
                continue
            if delta == 0 or delta > 1:
                continue
            enriched = dict(candidate)
            enriched["source"] = "neighbor_context"
            enriched["neighbor_distance"] = delta
            neighbors.append(enriched)

        neighbors.sort(
            key=lambda item: (
                -_chunk_equation_count(item) if math_priority else 0,
                item.get("neighbor_distance", 99),
                item.get("chunk_id", ""),
            )
        )

        for neighbor in neighbors[:max_neighbors_per_chunk]:
            expanded.append(neighbor)
            seen_ids.add(str(neighbor.get("chunk_id", "")))

    return expanded

# =========================
# Web fallback
# =========================

# Strict macro authorities for international data metrics
INTERNATIONAL_DOMAINS = [
    "worldbank.org", 
    "imf.org", 
    "fred.stlouisfed.org", 
    "oecd.org", 
    "ourworldindata.org"
]

# Regional authority filters for Pakistan economic data tracking
PAKISTAN_DOMAINS = [
    "sbp.org.pk",       # State Bank of Pakistan
    "pbs.gov.pk",       # Pakistan Bureau of Statistics
    "finance.gov.pk",   # Ministry of Finance Pakistan
    "pc.gov.pk"         # Planning Commission of Pakistan
]

def _normalize_markdown(text: str) -> str:
    """Cleans unicode debris but preserves layout breaks and table columns intact."""
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\u200b", " ").replace("\ufeff", " ")
    return text.strip()

def _trusted_web_domains(query: str) -> List[str]:
    q_lower = query.lower()
    is_pakistan = any(word in q_lower for word in ["pakistan", "pk", "rupee", "pkr", "sbp", "pbs", "lahore", "lums"])
    if is_pakistan:
        return PAKISTAN_DOMAINS + ["worldbank.org", "imf.org"]
    return INTERNATIONAL_DOMAINS


def _url_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().removeprefix("www.")
    except Exception:
        return ""


def _is_trusted_url(url: str, trusted_domains: Sequence[str]) -> bool:
    domain = _url_domain(url)
    return any(domain == trusted or domain.endswith("." + trusted) for trusted in trusted_domains)


def _extract_page_text(url: str, fallback: str = "", max_chars: int = 6000) -> str:
    if BeautifulSoup is None:
        return _normalize_markdown(fallback)[:max_chars]

    try:
        response = requests.get(
            url,
            timeout=12,
            headers={"User-Agent": "Mozilla/5.0 MacroAssist/1.0"},
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            tag.decompose()
        main = soup.find("main") or soup.find("article") or soup.body or soup
        text = main.get_text("\n", strip=True)
        text = _normalize_markdown(text)
        return (text or fallback)[:max_chars]
    except Exception:
        return _normalize_markdown(fallback)[:max_chars]


def search_web_fallback(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Search trusted economics domains through DuckDuckGo and extract page text locally.
    No Firecrawl or API key is required.
    """
    if DDGS is None:
        print("[WEB FALLBACK ERROR] duckduckgo-search is not installed.")
        return []

    trusted_domains = _trusted_web_domains(query)
    domain_filter = " OR ".join(f"site:{domain}" for domain in trusted_domains)
    search_query = f"({domain_filter}) {query}"

    try:
        items: List[Dict[str, str]] = []
        with DDGS() as ddgs:
            results = ddgs.text(search_query, max_results=max_results * 3)
            for result in results:
                href = (result.get("href") or result.get("url") or "").strip()
                if not href or not _is_trusted_url(href, trusted_domains):
                    continue
                title = _normalize_text(result.get("title", "Untitled Source"))
                fallback = result.get("body", "") or result.get("snippet", "")
                snippet = _extract_page_text(href, fallback=fallback)
                if not snippet:
                    continue
                items.append({
                    "title": title,
                    "url": href,
                    "snippet": snippet,
                })
                if len(items) >= max_results:
                    break
        return items

    except Exception as exc:
        print(f"[WEB FALLBACK ERROR] {exc}")
        return []


def format_web_context(results: List[Dict[str, str]]) -> str:
    if not results:
        return ""
    parts = []
    for idx, item in enumerate(results, start=1):
        title = _normalize_text(item.get("title", ""))
        url = _normalize_text(item.get("url", ""))
        # Bypasses space collapsing to keep layout structure valid for generation step
        snippet = _normalize_markdown(item.get("snippet", "")) 
        parts.append(
            f"""<web_source id="{idx}">
            <title>{title}</title>
            <url>{url}</url>
            <content>{snippet}</content>
            </web_source>"""
        )
    return "<web_context>\n" + "\n\n".join(parts) + "\n</web_context>"
# =========================
# Context formatting and generation
# =========================

def _extract_citation_label(chunk: Dict[str, Any]) -> str:
    book_title = _safe_label(chunk.get("book_title"), "")
    section = _safe_label(chunk.get("section"), "")
    chapter = _safe_label(chunk.get("chapter"), "")
    if book_title and book_title != "Unknown":
        if chapter and chapter != "Unknown" and section and section != "Unknown":
            return f"{book_title} | {chapter} | {section}"
        if section and section != "Unknown":
            return f"{book_title} | {section}"
        return book_title
    if section and section != "Unknown":
        return section
    return "Retrieved Context"


def _deduplicate_by_section(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique = []
    for chunk in chunks:
        key = (
            _safe_label(chunk.get("book_title"), "unknown").lower(),
            _safe_label(chunk.get("section"), "unknown").lower(),
            _safe_label(chunk.get("chapter"), "unknown").lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(chunk)
    return unique


def _filter_chunks_by_confidence(
    reranked_chunks: List[Dict[str, Any]],
    min_cross_encoder_score: float = 0.0,
) -> List[Dict[str, Any]]:
    return [
        chunk for chunk in reranked_chunks
        if chunk.get("cross_encoder_score") is not None and chunk.get("cross_encoder_score") >= min_cross_encoder_score
    ]


def _format_textbook_context(
    reranked_chunks: List[Dict[str, Any]],
    max_chunks: int = 3,
    max_chars_per_chunk: int = 2200,
) -> str:
    if not reranked_chunks:
        return ""

    selected = reranked_chunks[:max_chunks]
    parts = []
    for idx, chunk in enumerate(selected, start=1):
        citation_label = _extract_citation_label(chunk)
        book_title = _safe_label(chunk.get("book_title"), "Unknown Book")
        chapter = _safe_label(chunk.get("chapter"), "Unknown Chapter")
        section = _safe_label(chunk.get("section"), "Unknown Section")
        text = _normalize_text(chunk.get("text", ""))

        if len(text) > max_chars_per_chunk:
            text = text[:max_chars_per_chunk].rsplit(" ", 1)[0].strip() + " ..."

        parts.append(
            f"""<textbook_source id="{idx}">
            <citation>{citation_label}</citation>
            <book>{book_title}</book>
            <chapter>{chapter}</chapter>
            <section>{section}</section>
            <content>{text}</content>
            </textbook_source>"""
        )

    return "<textbook_context>\n" + "\n\n".join(parts) + "\n</textbook_context>"


def _build_generation_prompt(
    raw_query: str,
    optimized_query: str,
    textbook_context: str,
    web_context: str,
    intent_signals: Optional[IntentSignals] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    prompt_parts = [
        f"User question:\n{raw_query}",
        f"Optimized query:\n{optimized_query}",
    ]
    if intent_signals:
        prompt_parts.append(f"Intent task:\n{intent_signals.task}")

    if chat_history:
        recent = chat_history[-6:]
        history_lines = []
        for turn in recent:
            role = _normalize_text(turn.get("role", "")).lower()
            content = _normalize_text(turn.get("content", ""))
            if not content:
                continue
            history_lines.append(f"{role.title()}: {content}")
        if history_lines:
            prompt_parts.append("Conversation history:\n" + "\n".join(history_lines))

    if textbook_context:
        prompt_parts.append(f"Retrieved textbook context:\n{textbook_context}")
    if web_context:
        prompt_parts.append(f"Retrieved web context:\n{web_context}")

    return "\n\n".join(prompt_parts)


def generate_response_from_context(
    raw_query: str,
    optimized_query: str,
    reranked_chunks: List[Dict[str, Any]],
    web_results: List[Dict[str, str]],
    bm25_bundle: Optional[Dict[str, Any]] = None,
    intent_signals: Optional[IntentSignals] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
    provider: str = "Gemini",
    model_name: str = "gemini-2.5-pro",
    max_context_chunks: int = 3,
    min_cross_encoder_score: float = 0.0,
    return_metadata: bool = False,
) -> str | Tuple[str, Dict[str, Any]]:
    raw_query = _normalize_text(raw_query)
    optimized_query = _normalize_text(optimized_query)

    filtered_chunks = _filter_chunks_by_confidence(reranked_chunks, min_cross_encoder_score=min_cross_encoder_score)
    filtered_chunks = _deduplicate_by_section(filtered_chunks)
    filtered_chunks = sorted(filtered_chunks, key=lambda x: x.get("cross_encoder_score", float("-inf")), reverse=True)
    filtered_chunks = expand_with_neighbor_chunks(filtered_chunks, bm25_bundle or {}, intent_signals)

    textbook_context = _format_textbook_context(filtered_chunks, max_chunks=max_context_chunks)
    web_context = format_web_context(web_results)

    if not textbook_context and not web_context:
        meta = {
            "status": "no_retrieval_confidence",
            "intent": intent_signals.task if intent_signals else "unknown",
            "provider": provider,
            "model_name": model_name,
            "raw_query": raw_query,
            "optimized_query": optimized_query,
            "used_textbook_sources": [],
            "used_web_sources": [],
        }
        return (NO_INFO_RESPONSE, meta) if return_metadata else NO_INFO_RESPONSE

    prompt = _build_generation_prompt(
        raw_query=raw_query,
        optimized_query=optimized_query,
        textbook_context=textbook_context,
        web_context=web_context,
        intent_signals=intent_signals,
        chat_history=chat_history,
    )

    try:
        answer = _call_text_llm(
            provider=provider,
            model_name=model_name,
            system_prompt=_system_prompt_for_intent(intent_signals, used_web=bool(web_results)),
            user_prompt=prompt,
            temperature=0.2,
            top_p=0.95,
            max_output_tokens=2000,
        )
        answer = answer.strip()
        if not answer:
            answer = NO_INFO_RESPONSE

        used_textbook_sources = []
        for chunk in filtered_chunks[:max_context_chunks]:
            used_textbook_sources.append(
                {
                    "citation": _extract_citation_label(chunk),
                    "book_title": _safe_label(chunk.get("book_title"), ""),
                    "chapter": _safe_label(chunk.get("chapter"), ""),
                    "section": _safe_label(chunk.get("section"), ""),
                    "chunk_id": chunk.get("chunk_id"),
                    "cross_encoder_score": chunk.get("cross_encoder_score"),
                    "rrf_score": chunk.get("rrf_score"),
                }
            )

        used_web_sources = []
        for item in web_results:
            used_web_sources.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("snippet", ""),
                }
            )

        meta = {
            "status": "ok",
            "intent": intent_signals.task if intent_signals else "unknown",
            "provider": provider,
            "model_name": model_name,
            "raw_query": raw_query,
            "optimized_query": optimized_query,
            "used_textbook_sources": used_textbook_sources,
            "used_web_sources": used_web_sources,
        }
        return (answer, meta) if return_metadata else answer

    except Exception as exc:
        meta = {
            "status": "error",
            "intent": intent_signals.task if intent_signals else "unknown",
            "provider": provider,
            "model_name": model_name,
            "raw_query": raw_query,
            "optimized_query": optimized_query,
            "error": str(exc),
            "used_textbook_sources": [],
            "used_web_sources": [],
        }
        return (NO_INFO_RESPONSE, meta) if return_metadata else NO_INFO_RESPONSE


def final_rag_pipeline(
    raw_query: str,
    collection,
    bm25_bundle: Dict[str, Any],
    equation_index: Dict[str, Any],
    dense_model: SentenceTransformer,
    reranker_model: CrossEncoder,
    chat_history: Optional[List[Dict[str, str]]] = None,
    retrieval_top_k: int = 5,
    query_provider: str = "Gemini",
    query_model_name: str = "gemini-2.5-flash",
    generation_provider: str = "Gemini",
    generation_model_name: str = "gemini-2.5-pro",
    min_cross_encoder_score: float = 0.0,
    web_enabled: bool = True,
    web_fallback_threshold: float = WEB_FALLBACK_THRESHOLD_DEFAULT,
    return_metadata: bool = False,
):
    optimized_query = optimize_and_expand_query(
        raw_query,
        provider=query_provider,
        model_name=query_model_name,
        return_metadata=False,
    )
    retrieval_query = build_retrieval_query(raw_query, optimized_query)

    reranked_chunks = final_pipeline_retrieval(
        retrieval_query,
        collection=collection,
        bm25_bundle=bm25_bundle,
        dense_model=dense_model,
        reranker_model=reranker_model,
        dense_top_k=12,
        sparse_top_k=12,
        fused_top_k=15,
        rerank_top_k=8,
    )

    initial_intent = classify_intent(retrieval_query, reranked_chunks)
    equation_matches: List[Dict[str, Any]] = []
    if initial_intent.is_numerical or initial_intent.wants_formula:
        equation_matches = search_equations(retrieval_query, equation_index, dense_model, top_k=8)
        reranked_chunks = _equation_matches_to_chunks(equation_matches) + reranked_chunks

    intent_signals = classify_intent(retrieval_query, reranked_chunks)
    top_textbook_score = next(
        (
            chunk.get("cross_encoder_score")
            for chunk in reranked_chunks
            if chunk.get("source") != "equation_index" and chunk.get("cross_encoder_score") is not None
        ),
        None,
    )
    top_score = top_textbook_score
    econ_related = intent_signals.is_macro

    # Numerical math path: only for macro-style calculation questions.
    if intent_signals.is_numerical:
        solver_result = solve_macro_numerical_problem(raw_query, reranked_chunks, equation_index, dense_model)

        if solver_result is not None:
            values_text = ", ".join(
                f"{k}={v:g}" for k, v in solver_result["values"].items()
            )

            source_chunks = solver_result.get("source_chunks", [])
            source_labels = []
            for source_chunk in source_chunks:
                label = _extract_citation_label(source_chunk)
                if label not in source_labels:
                    source_labels.append(label)
            if not source_labels:
                source_labels = ["Canonical macro formula"]
            source_lines = "\n".join(f"- {label}" for label in source_labels[:8])

            answer = f"""# Direct Answer

                    {solver_result["result"]:g}

                    ## Formula Used

                    $$
                    {solver_result["formula"]}
                    $$

                    ## Substituted Values

                    {values_text}

                    ## Solver

                    {solver_result.get("method", "deterministic")}

                    ## Sources

                    {source_lines}
                    """

            meta = {
                "status": "solved_deterministically",
                "intent": intent_signals.task,
                "raw_query": raw_query,
                "optimized_query": optimized_query,
                "retrieval_query": retrieval_query,
                "econ_related": econ_related,
                "top_cross_encoder_score": top_score,
                "solver_used": True,
                "solver_result": solver_result,
                "used_textbook_sources": [
                    {
                        "citation": _extract_citation_label(source_chunk),
                        "book_title": _safe_label(source_chunk.get("book_title"), ""),
                        "chapter": _safe_label(source_chunk.get("chapter"), ""),
                        "section": _safe_label(source_chunk.get("section"), ""),
                        "chunk_id": source_chunk.get("chunk_id"),
                        "cross_encoder_score": source_chunk.get("cross_encoder_score"),
                        "rrf_score": source_chunk.get("rrf_score"),
                    }
                    for source_chunk in source_chunks
                ],
                "used_web_sources": [],
            }
            return (answer, meta) if return_metadata else answer

        # ------------
        calc_prompt = f"""
        User question:

        {raw_query}

        This is a macroeconomics calculation problem.

        Use ONLY the information provided in the question.
        Show all calculations step-by-step.
        Do not require textbook evidence.
        Do not refuse because textbook retrieval is missing.
        If sufficient numerical information is present, solve the problem directly.
        """.strip()

        try:
            answer = (
                _call_text_llm(
                    provider=generation_provider,
                    model_name=generation_model_name,
                    system_prompt="You are a macroeconomics numerical problem solver.",
                    user_prompt=calc_prompt,
                    temperature=0.0,
                    top_p=0.95,
                    max_output_tokens=2000,
                ).strip()
                or NO_INFO_RESPONSE
            )

            meta = {
                "status": "generated_for_calculation",
                "intent": intent_signals.task,
                "raw_query": raw_query,
                "optimized_query": optimized_query,
                "retrieval_query": retrieval_query,
                "econ_related": econ_related,
                "top_cross_encoder_score": top_score,
                "solver_used": False,
                "used_textbook_sources": [],
                "used_web_sources": [],
                "used_web_fallback": False,
            }

            return (answer, meta) if return_metadata else answer

        except Exception as exc:
            meta = {
                "status": "calculation_error",
                "intent": intent_signals.task,
                "raw_query": raw_query,
                "optimized_query": optimized_query,
                "error": str(exc),
            }

            return (NO_INFO_RESPONSE, meta) if return_metadata else NO_INFO_RESPONSE
    
    

    use_web = False
    web_results: List[Dict[str, str]] = []

    if web_enabled and econ_related and intent_signals.wants_web:
        if top_score is None or top_score < web_fallback_threshold:
            use_web = True
            web_results = search_web_fallback(retrieval_query, max_results=5)

    if intent_signals.task == "formula" and not equation_matches and (top_score is None or top_score < web_fallback_threshold):
        answer = NO_INFO_RESPONSE
        meta = {
            "status": "no_formula_match",
            "intent": intent_signals.task,
            "raw_query": raw_query,
            "optimized_query": optimized_query,
            "retrieval_query": retrieval_query,
            "econ_related": econ_related,
            "top_cross_encoder_score": top_score,
            "used_web_fallback": False,
            "used_textbook_sources": [],
            "used_web_sources": [],
        }
        return (answer, meta) if return_metadata else answer

    # If the query is not economics-related and retrieval is weak, do not use web fallback.
    if not econ_related and (top_score is None or top_score < web_fallback_threshold):
        answer = NO_INFO_RESPONSE
        meta = {
            "status": "no_retrieval_confidence",
            "intent": intent_signals.task,
            "raw_query": raw_query,
            "optimized_query": optimized_query,
            "retrieval_query": retrieval_query,
            "econ_related": econ_related,
            "top_cross_encoder_score": top_score,
            "used_web_fallback": False,
            "used_textbook_sources": [],
            "used_web_sources": [],
        }
        return (answer, meta) if return_metadata else answer

    response = generate_response_from_context(
        raw_query=raw_query,
        optimized_query=optimized_query,
        reranked_chunks=reranked_chunks,
        web_results=web_results if use_web else [],
        bm25_bundle=bm25_bundle,
        intent_signals=intent_signals,
        chat_history=chat_history,
        provider=generation_provider,
        model_name=generation_model_name,
        max_context_chunks=retrieval_top_k,
        min_cross_encoder_score=min_cross_encoder_score,
        return_metadata=return_metadata,
    )
    if return_metadata:
        answer, meta = response
        meta.update({
            "retrieval_query": retrieval_query,
            "econ_related": econ_related,
            "top_cross_encoder_score": top_score,
            "used_web_fallback": use_web,
            "equation_matches": equation_matches,
        })
        if use_web and not web_results:
            meta["status"] = "no_safe_web_result"
        return answer, meta
    return response


# =========================
# Sidebar
# =========================

with st.sidebar:
    st.header("Settings")

    paths = get_paths()

    st.subheader("Query optimizer")
    query_provider = st.selectbox("Query provider", options=LLM_PROVIDER_OPTIONS, index=0)
    query_model_options = {
        "Gemini": ["gemini-2.5-flash", "gemini-3.1-flash-lite"],
        "OpenAI": ["gpt-4.1-mini", "gpt-5.4-mini", "gpt-5.4-nano"],
        "Groq": ["openai/gpt-oss-20b"],
        "DeepSeek": ["deepseek-v4-flash"],
    }
    query_model_name = st.selectbox("Query model", options=query_model_options[query_provider], index=0)

    st.subheader("Answer generation")
    generation_provider = st.selectbox("Generation provider", options=LLM_PROVIDER_OPTIONS, index=0)
    generation_model_options = {
        "Gemini": ["gemini-2.5-flash", "gemini-3.1-flash-lite", "gemini-3.5-flash"],
        "OpenAI": ["gpt-4.1-mini", "gpt-5.4-mini"],
        "Groq": ["openai/gpt-oss-120b"],
        "DeepSeek": ["deepseek-v4-flash", "deepseek-v4-pro"],
    }
    generation_model_name = st.selectbox("Generation model", options=generation_model_options[generation_provider], index=0)

    st.subheader("Retrieval")
    retrieval_top_k = st.slider("Top context chunks", min_value=1, max_value=5, value=3, step=1)
    min_ce_score = st.slider("Minimum reranker score", min_value=-2.0, max_value=2.0, value=0.0, step=0.05)
    web_enabled = st.checkbox("Enable web fallback", value=True)
    web_fallback_threshold = st.slider(
        "Web fallback threshold",
        min_value=-0.5,
        max_value=1.0,
        value=WEB_FALLBACK_THRESHOLD_DEFAULT,
        step=0.01,
    )

    st.divider()
    if st.button("Refresh index status", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# =========================
# Session state
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================
# Index status
# =========================
# --------------------------

manifest = load_manifest_cached(paths["manifest_path"])
bm25_bundle = load_bm25_bundle_cached(paths["bm25_path"])

index_exists = _collection_exists(paths["persist_dir"], paths["collection_name"])
collection = None
index_error = None

if index_exists:
    try:
        collection = load_collection_cached(paths["persist_dir"], paths["collection_name"])
    except Exception as exc:
        index_error = str(exc)

index_ready = index_exists and collection is not None and bool(bm25_bundle)

if not index_ready:
    st.warning("No ready macroeconomics index found.")
    st.info(
        "Run ingest_macro_books.py first to build the persistent ChromaDB index, then reopen this app."
    )
    if manifest:
        st.subheader("Last saved manifest")
        st.json(manifest)
    if index_error:
        st.error(index_error)
else:
    total_books = manifest.get("total_books", "?")
    total_chunks = manifest.get("total_chunks", "?")
    st.success(f"Index ready: {total_books} books | {total_chunks} chunks")

    with st.expander("Index summary", expanded=False):
        st.json(manifest)

    dense_model, reranker_model, equation_index = load_models(
        paths["embedding_model_name"],
        paths["reranker_model_name"],
        paths["equation_index_path"],
    )

    with st.expander("Equation index summary", expanded=False):
        st.json({
            "path": paths["equation_index_path"],
            "grouped_formula_count": len(equation_index.get("formulas", [])),
        })

    # =========================
    # Chat history
    # =========================
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("metadata"):
                with st.expander("Pipeline telemetry"):
                    st.json(msg["metadata"])

    # =========================
    # Chat input
    # =========================
    if prompt := st.chat_input("Ask a macroeconomics question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching textbooks and generating answer..."):
                answer, metadata = final_rag_pipeline(
                    raw_query=prompt,
                    collection=collection,
                    bm25_bundle=bm25_bundle,
                    equation_index=equation_index,
                    dense_model=dense_model,
                    reranker_model=reranker_model,
                    chat_history=st.session_state.messages,
                    retrieval_top_k=retrieval_top_k,
                    query_provider=query_provider,
                    query_model_name=query_model_name,
                    generation_provider=generation_provider,
                    generation_model_name=generation_model_name,
                    min_cross_encoder_score=min_ce_score,
                    web_enabled=web_enabled,
                    web_fallback_threshold=web_fallback_threshold,
                    return_metadata=True,
                )

            # st.markdown(answer)
            answer = _standardize_equation_delimiters(answer)
            _render_answer_with_equations(answer)
            if metadata:
                with st.expander("Pipeline telemetry"):
                    st.json(metadata)

        st.session_state.messages.append({"role": "assistant", "content": answer, "metadata": metadata})
