import ast
import json
import math
import operator
import os
import re
import time
import unicodedata

from collections import deque
from itertools import zip_longest
from pathlib import Path

import chromadb
import numpy as np
import pandas as pd

from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from rank_bm25 import BM25Okapi


# ==============================
# Project Paths
# ==============================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"
CHROMA_DIR = BASE_DIR / "chroma_db"


# ==============================
# Environment Variables
# ==============================

load_dotenv(BASE_DIR / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is missing. Add it to the .env file.")


# ==============================
# Groq Configuration
# ==============================

groq_client = Groq(api_key=GROQ_API_KEY, timeout=60, max_retries=0)

MODEL = "qwen/qwen3.8-27b"
HELPER_MODEL = "openai/gpt-oss-20b"

MAX_WAIT = 90

LLM_STATS = {"calls": 0, "tokens": 0, "log": []}

# Tokens-per-minute limit of the primary model (see Groq console -> Limits).
# Put GROQ_TPM_LIMIT=<number> in .env. Empty or 0 disables throttling.
TPM_LIMIT = int(os.getenv("GROQ_TPM_LIMIT", "0") or 0) or None
_window = deque()

# Models that rejected the reasoning parameter (avoid repeating the 400 error)
_NO_REASONING = set()


def _rk(model):
    m = model or MODEL
    if m in _NO_REASONING:
        return {}
    if "gpt-oss" in m:
        return {"reasoning_effort": "low"}
    if "qwen" in m:
        return {"reasoning_effort": "none"}
    return {}


def _retry_after(e):
    try:
        h = e.response.headers
        return float(h.get("retry-after") or h.get("Retry-After"))
    except Exception:
        return None


def _throttle(messages):
    if not TPM_LIMIT:
        return

    est = len(json.dumps(messages)) // 3 + 500

    while True:
        now = time.time()

        while _window and now - _window[0][0] > 60:
            _window.popleft()

        if sum(t for _, t in _window) + est <= TPM_LIMIT:
            return

        if not _window:
            return

        wait = 60 - (now - _window[0][0]) + 0.5
        time.sleep(min(5, max(0.5, wait)))


def llm_chat(messages, tools=None, effort="low", model=None):
    used_model = model or MODEL
    kwargs = dict(model=used_model, messages=messages)

    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    for attempt in range(5):
        _throttle(messages)
        t0 = time.time()

        try:
            try:
                r = groq_client.chat.completions.create(
                    **_rk(used_model), **kwargs
                )
            except Exception as e:
                if (
                    getattr(e, "status_code", None) == 400
                    and "reasoning" in str(e).lower()
                ):
                    _NO_REASONING.add(used_model)
                    r = groq_client.chat.completions.create(**kwargs)
                else:
                    raise

            tokens = r.usage.total_tokens if r.usage else 0

            _window.append((time.time(), tokens))

            LLM_STATS["calls"] += 1
            LLM_STATS["tokens"] += tokens
            LLM_STATS["log"].append((round(time.time() - t0, 1), tokens))

            msg = r.choices[0].message
            msg.content = re.sub(
                r"<think>.*?</think>", "", msg.content or "", flags=re.S
            ).strip()
            return msg

        except Exception as e:
            code = getattr(e, "status_code", None)

            if (
                code == 400
                and tools is None
                and "tool" in str(e).lower()
                and attempt < 2
            ):
                messages = messages + [{
                    "role": "system",
                    "content": "Answer in plain text only. Do not call any tool or function."
                }]
                kwargs["messages"] = messages
                continue

            if code in (429, 500, 503) and attempt < 4:
                wait = (_retry_after(e) or 8) + 1

                if wait > MAX_WAIT:
                    raise RuntimeError(
                        f"Groq rate limit reached (about {wait / 60:.0f} min wait). "
                        f"Try again later. Details: {str(e)[:250]}"
                    )

                print(f"  [LLM error {code}, waiting {wait:.0f}s]")
                time.sleep(wait)

            else:
                raise


def llm_text(prompt, model=None):
    return (
        llm_chat([{"role": "user", "content": prompt}], model=model).content or ""
    ).strip()


# ==============================
# Load Economic Data (CSV Files)
# ==============================

CSV_FILES = {
    "gdp_current_usd": "gdp_current_usd.csv",
    "gdp_growth_pct": "gdp_growth_pct.csv",
    "inflation_pct": "inflation_pct.csv",
    "population": "population.csv",
    "unemployment_pct": "unemployment_pct.csv",
}

INDICATOR_MAP = {
    "gdp_current_usd": ("GDP", "current US$"),
    "gdp_growth_pct": ("GDP growth", "annual %"),
    "inflation_pct": ("Inflation (consumer prices)", "annual %"),
    "population": ("Population", "people"),
    "unemployment_pct": ("Unemployment", "% of labor force"),
}


def load_indicator_table():
    parts = []

    for table_name, file_name in CSV_FILES.items():
        file_path = DATA_DIR / file_name

        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        df = pd.read_csv(file_path, encoding="utf-8-sig", skiprows=4)

        if df.empty:
            raise ValueError(f"Data file is empty: {file_name}")

        indicator, unit = INDICATOR_MAP[table_name]
        year_columns = [c for c in df.columns if str(c).isdigit()]

        long_df = df.melt(
            id_vars=["Country Name"],
            value_vars=year_columns,
            var_name="year",
            value_name="value",
        ).rename(columns={"Country Name": "country"})

        long_df["indicator"] = indicator
        long_df["unit"] = unit
        long_df["year"] = pd.to_numeric(long_df["year"], errors="coerce")
        long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
        long_df = long_df.dropna(subset=["country", "year", "value"])

        parts.append(long_df)

    out = pd.concat(parts, ignore_index=True)
    out["year"] = out["year"].astype(int)
    return out


table = load_indicator_table()
print("Prepared table:", table.shape)


# ==============================
# PDF Reports
# ==============================

PDF_FILES = {
    "world_bank_2021": "egypt_economic_monitor_dec_2021.pdf",
    "world_bank_2022": "egypt_economic_monitor_dec_2022.pdf",
    "imf_2025": "imf_egypt_country_report_25_186_jul_2025.pdf",
}

PUBLISHER = {
    "egypt_economic_monitor_dec_2021.pdf": "World Bank",
    "egypt_economic_monitor_dec_2022.pdf": "World Bank",
    "imf_egypt_country_report_25_186_jul_2025.pdf": "IMF",
}

SOURCE_TITLES = {
    "egypt_economic_monitor_dec_2021.pdf": "World Bank, Egypt Economic Monitor (December 2021)",
    "egypt_economic_monitor_dec_2022.pdf": "World Bank, Egypt Economic Monitor (December 2022)",
    "imf_egypt_country_report_25_186_jul_2025.pdf": "IMF Country Report No. 25/186 (July 2025)",
}

SOURCE_META = {
    "egypt_economic_monitor_dec_2021.pdf": {"published": "December 2021", "covers_until": 2021},
    "egypt_economic_monitor_dec_2022.pdf": {"published": "December 2022", "covers_until": 2022},
    "imf_egypt_country_report_25_186_jul_2025.pdf": {"published": "July 2025", "covers_until": 2025},
}


def extract_report_pages():
    report_pages = {}

    for _, file_name in PDF_FILES.items():
        file_path = REPORTS_DIR / file_name

        if not file_path.exists():
            raise FileNotFoundError(f"PDF report not found: {file_path}")

        reader = PdfReader(str(file_path))
        pages = []

        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text()
            if page_text and page_text.strip():
                pages.append((page_number, page_text.strip()))

        report_pages[file_name] = pages
        print(f"{file_name}: {len(reader.pages)} pages, {len(pages)} with text")

    return report_pages


report_pages = extract_report_pages()


# ==============================
# Text Cleaning and Chunking
# ==============================

def clean_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

chunks = []

for f, pages in report_pages.items():
    for num, text in pages:
        cleaned = clean_text(text)

        if len(cleaned) < 50:
            continue

        for piece in splitter.split_text(cleaned):
            if len(piece) >= 80:
                chunks.append({
                    "id": str(len(chunks)),
                    "text": piece,
                    "source": f,
                    "publisher": PUBLISHER[f],
                    "page": num,
                    "location": f"page {num}",
                })

chunks_df = pd.DataFrame(chunks)
print("Total chunks:", len(chunks_df))


# ==============================
# Embeddings + ChromaDB
# ==============================

ef = embedding_functions.DefaultEmbeddingFunction()

texts = chunks_df["text"].tolist()

db = chromadb.PersistentClient(path=str(CHROMA_DIR))

collection = db.get_or_create_collection(
    "reports", metadata={"hnsw:space": "cosine"}
)

existing_ids = set(collection.get(include=[])["ids"])
expected_ids = set(chunks_df["id"].tolist())

if existing_ids != expected_ids:
    # Embeddings are computed only when the collection must be rebuilt
    embeddings = []

    for i in range(0, len(texts), 256):
        embeddings.extend(ef(texts[i:i + 256]))
        print(f"Embedded {min(i + 256, len(texts))}/{len(texts)}")

    db.delete_collection("reports")
    collection = db.create_collection(
        "reports", metadata={"hnsw:space": "cosine"}
    )

    metas = chunks_df[["source", "publisher", "page", "location"]].to_dict("records")

    for i in range(0, len(texts), 500):
        collection.add(
            ids=chunks_df["id"].tolist()[i:i + 500],
            documents=texts[i:i + 500],
            embeddings=[np.array(e).tolist() for e in embeddings[i:i + 500]],
            metadatas=metas[i:i + 500],
        )

    print("Collection rebuilt.")
else:
    print("Existing collection reused.")

print("Stored in DB:", collection.count())


# ==============================
# BM25
# ==============================

STOP = {
    "the", "a", "an", "is", "are", "was", "of", "to", "in", "on", "and",
    "or", "for", "what", "how", "why", "do", "does", "we", "i", "it",
    "this", "that", "with", "as", "by",
}


def tok(s):
    return [
        w for w in re.findall(r"[a-z0-9]+", s.lower())
        if w not in STOP and len(w) > 1
    ]


ALL_IDS = chunks_df["id"].tolist()
BM25 = BM25Okapi([tok(d) for d in texts])
ID2ROW = {cid: i for i, cid in enumerate(ALL_IDS)}


# ==============================
# Sources and Noise Filter
# ==============================

SOURCES = {}
CURRENT_QUESTION = ""


def register_source(**info):
    tag = f"S{len(SOURCES) + 1}"
    SOURCES[tag] = info
    return tag


def digit_ratio(t):
    chars = [c for c in t if not c.isspace()]
    return sum(c.isdigit() for c in chars) / max(1, len(chars))


MAX_DIGIT_RATIO = 0.25


def is_noise(t):
    axis_labels = len(re.findall(r"\b[A-Z][a-z]{2}-\d{2}\b", t)) >= 6
    toc = len(re.findall(r"(?:\.\s?){5,}", t)) >= 2
    return digit_ratio(t) > MAX_DIGIT_RATIO or axis_labels or toc


chunks_df["noise"] = chunks_df["text"].apply(is_noise)
print("Excluded chunks:", int(chunks_df["noise"].sum()))


# ==============================
# Hybrid Search
# ==============================

def hybrid_pool(query, source=None, publisher=None, n_vec=8, n_bm25=8):
    conds = []

    if source:
        conds.append({"source": source})
    if publisher:
        conds.append({"publisher": publisher})

    where = (
        None if not conds
        else conds[0] if len(conds) == 1
        else {"$and": conds}
    )

    q = [np.array(e).tolist() for e in ef([query])]

    vec_ids = list(
        collection.query(query_embeddings=q, n_results=n_vec, where=where)["ids"][0]
    )

    scores = BM25.get_scores(tok(query))
    mask = np.ones(len(chunks_df), dtype=bool)

    if source:
        mask &= chunks_df["source"].values == source
    if publisher:
        mask &= chunks_df["publisher"].values == publisher

    scores = np.where(mask, scores, 0)

    bm_ids = [
        ALL_IDS[i]
        for i in np.argsort(scores)[::-1][:n_bm25]
        if scores[i] > 0
    ]

    merged, seen = [], set()

    for pair in zip_longest(vec_ids, bm_ids):
        for cid in pair:
            if cid and cid not in seen:
                seen.add(cid)
                merged.append(cid)

    return [chunks_df.iloc[ID2ROW[c]] for c in merged]


BOILER = re.compile(
    r"\b(egypt economic monitor|economic monitor|country report|world bank|imf|the report|report|monitor|"
    r"december 2021|december 2022|july 2025|dec 2021|dec 2022)\b",
    re.I,
)


def clean_query(q):
    cq = re.sub(r"\s+", " ", BOILER.sub(" ", q)).strip()
    return cq if len(tok(cq)) >= 2 else q


def stem6(w):
    return w[:6]


def best_window(text, query, max_chars=450):
    sents = re.split(r"(?<=[.!?])\s+", text)
    qs = {stem6(w) for w in tok(query)}
    scores = [sum(stem6(w) in qs for w in set(tok(s))) for s in sents]

    best, bi, bj = -1, 0, 1

    for i in range(len(sents)):
        length, total = 0, 0

        for j in range(i, len(sents)):
            length += len(sents[j]) + 1

            if length > max_chars and j > i:
                break

            total += scores[j]

            if total > best:
                best, bi, bj = total, i, j + 1

    window = " ".join(sents[bi:bj])[:max_chars + 150]
    return ("... " if bi > 0 else "") + window


def search_documents(query, publisher=None, source=None, k=4):
    rows = [
        r for r in hybrid_pool(clean_query(query), source, publisher)
        if not r["noise"]
    ][:k]

    if not rows:
        return "No relevant passages found."

    years = {
        int(y)
        for y in re.findall(r"\b(20[12]\d)\b", query + " " + CURRENT_QUESTION)
    }

    late, out = set(), []

    for r in rows:
        meta = SOURCE_META[r["source"]]
        late.update(y for y in years if y > meta["covers_until"])
        title = SOURCE_TITLES[r["source"]]

        tag = register_source(
            kind="document",
            title=title,
            source=r["source"],
            location=r["location"],
            publisher=r["publisher"],
            text=r["text"],
        )

        shown = best_window(r["text"], clean_query(query) + " " + CURRENT_QUESTION)

        out.append(
            f"[{tag}] {title}, {r['location']} (published {meta['published']})\n{shown}"
        )

    warning = ""

    if late:
        published = ", ".join(
            sorted({SOURCE_META[r["source"]]["published"] for r in rows})
        )
        warning = (
            f"WARNING: the passages below come from reports published in {published}. "
            f"They cannot describe events in {sorted(late)}. Do not attribute those "
            "years to them. Search the IMF report or say clearly that this period "
            "is not covered by that source.\n\n"
        )

    return warning + "\n\n".join(out)


# ==============================
# Table Analysis Tool
# ==============================

IND_ALIASES = {
    "gdp growth": "GDP growth",
    "real gdp growth": "GDP growth",
    "growth": "GDP growth",
    "inflation": "Inflation (consumer prices)",
    "cpi": "Inflation (consumer prices)",
    "unemployment": "Unemployment",
    "gdp": "GDP",
    "population": "Population",
}

COUNTRY_ALIASES = {
    "egypt": "Egypt, Arab Rep.",
    "مصر": "Egypt, Arab Rep.",
    "turkey": "Turkiye",
    "turkiye": "Turkiye",
    "türkiye": "Turkiye",
}

ALL_COUNTRIES = {c.lower(): c for c in table["country"].unique()}


def resolve_indicator(name):
    key = name.strip().lower()

    if key in IND_ALIASES:
        return IND_ALIASES[key]

    for k, v in IND_ALIASES.items():
        if k in key:
            return v

    return None


def resolve_country(name):
    key = name.strip().lower()
    alias = COUNTRY_ALIASES.get(key)

    if alias and alias.lower() in ALL_COUNTRIES:
        return ALL_COUNTRIES[alias.lower()]

    return ALL_COUNTRIES.get(key)


def fmt(v, unit):
    return f"{v:,.0f}" if unit in ("current US$", "people") else f"{v:,.2f}"


OPERATIONS = [
    "values", "average", "min", "max", "change", "pct_change",
    "cagr", "rank", "rank_average", "latest",
]


def analyze_table(indicator, countries, start_year=None, end_year=None, operation="values"):
    ind = resolve_indicator(indicator)

    if ind is None:
        return (
            f"Unknown indicator '{indicator}'. Available: "
            "GDP growth, Inflation, Unemployment, GDP, Population."
        )

    names, unknown = [], []

    for c in countries:
        r = resolve_country(c)
        (names if r else unknown).append(r or c)

    if unknown:
        return (
            f"Unknown country: {unknown}. "
            "Use English country names, for example Egypt, Morocco, Turkey."
        )

    if operation not in OPERATIONS:
        return f"Unknown operation '{operation}'. Available: {OPERATIONS}"

    df = table[(table["indicator"] == ind) & (table["country"].isin(names))]

    if start_year is not None:
        df = df[df["year"] >= int(start_year)]
    if end_year is not None:
        df = df[df["year"] <= int(end_year)]

    if df.empty:
        return f"No data for {ind}, {names}, {start_year}-{end_year}."

    unit = df["unit"].iloc[0]
    wide = df.pivot(index="year", columns="country", values="value").sort_index()
    y0, y1 = int(wide.index.min()), int(wide.index.max())

    tag = register_source(
        kind="table",
        title="World Bank World Development Indicators",
        source=f"{ind} ({unit})",
        location=f"{', '.join(names)}, {y0}-{y1}",
    )

    head = f"[{tag}] World Bank WDI | {ind} ({unit}) | {y0}-{y1}\n"
    lines = []

    for c in wide.columns:
        s = wide[c].dropna()

        if s.empty:
            lines.append(f"{c}: no data in this range")
            continue

        a, b = int(s.index[0]), int(s.index[-1])

        if operation == "values":
            lines.append(
                f"{c}: " + ", ".join(f"{int(y)}={fmt(v, unit)}" for y, v in s.items())
            )

        elif operation == "average":
            lines.append(f"{c}: average {fmt(s.mean(), unit)} (years {a}-{b}, n={len(s)})")

        elif operation == "min":
            lines.append(f"{c}: min {fmt(s.min(), unit)} in {int(s.idxmin())}")

        elif operation == "max":
            lines.append(f"{c}: max {fmt(s.max(), unit)} in {int(s.idxmax())}")

        elif operation == "change":
            first_value, last_value = float(s.iloc[0]), float(s.iloc[-1])
            difference = round(last_value - first_value, 2)
            lines.append(
                f"{c}: {a}={fmt(first_value, unit)} -> {b}={fmt(last_value, unit)}, "
                f"change {fmt(difference, unit)}"
            )

        elif operation == "pct_change":
            first_value, last_value = float(s.iloc[0]), float(s.iloc[-1])

            if first_value != 0:
                pct = (last_value / first_value - 1) * 100
                lines.append(f"{c}: {a} -> {b}: {pct:,.2f}%")
            else:
                lines.append(f"{c}: percentage change not defined (starting value is zero)")

        elif operation == "cagr":
            if b > a and s.iloc[0] > 0 and s.iloc[-1] > 0:
                cagr = ((s.iloc[-1] / s.iloc[0]) ** (1 / (b - a)) - 1) * 100
                lines.append(f"{c}: CAGR {a}-{b} = {cagr:,.2f}% per year")
            else:
                lines.append(f"{c}: CAGR not defined for this range")

        elif operation == "latest":
            lines.append(f"{c}: latest year {b} = {fmt(s.iloc[-1], unit)}")

    if operation == "rank":
        last = {
            c: (int(wide[c].dropna().index[-1]), wide[c].dropna().iloc[-1])
            for c in wide.columns
            if wide[c].notna().any()
        }
        ranked = sorted(last.items(), key=lambda kv: kv[1][1], reverse=True)
        lines = [
            f"{i + 1}. {c}: {fmt(v, unit)} (year {y})"
            for i, (c, (y, v)) in enumerate(ranked)
        ]

    elif operation == "rank_average":
        means = wide.mean().dropna().sort_values(ascending=False)
        lines = [
            f"{i + 1}. {c}: {fmt(v, unit)} (average {y0}-{y1})"
            for i, (c, v) in enumerate(means.items())
        ]

    return head + "\n".join(lines)


# ==============================
# Safe Calculator Tool
# ==============================

_BIN = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}

_UN = {ast.USub: operator.neg, ast.UAdd: operator.pos}

_FUNCS = {
    "round": round, "abs": abs, "min": min, "max": max,
    "sqrt": math.sqrt, "log": math.log,
}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)

    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    ):
        return node.value

    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        left, right = _eval(node.left), _eval(node.right)

        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")

        return _BIN[type(node.op)](left, right)

    if isinstance(node, ast.UnaryOp) and type(node.op) in _UN:
        return _UN[type(node.op)](_eval(node.operand))

    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCS
        and not node.keywords
    ):
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])

    raise ValueError("unsupported expression")


def calculate(expression):
    try:
        result = _eval(ast.parse(expression.replace("^", "**"), mode="eval"))
        return f"{expression} = {round(result, 6)}"

    except ZeroDivisionError:
        return "Error: division by zero"

    except Exception as e:
        return (
            f"Error: {e}. Use plain numbers with "
            "+ - * / ** ( ) and round, abs, min, max, sqrt, log."
        )


# ==============================
# Document Summarization Tool
# ==============================

def summarize_document(source, focus):
    if source not in SOURCE_TITLES:
        return f"Unknown source. Available: {list(SOURCE_TITLES)}"

    rows = [
        r for r in hybrid_pool(focus, source=source, n_vec=10, n_bm25=6)
        if not r["noise"]
    ][:8]

    if not rows:
        return "No relevant passages found in this document."

    ctx = "\n\n".join(f"(page {r['page']}) {r['text']}" for r in rows)

    summary = llm_text(
        f"Summarize what the passages below say about: {focus}\n"
        f"Use only the passages. Maximum 120 words. "
        f"Mention page numbers like (p. 9).\n\n{ctx}"
    )

    pages = sorted({int(r["page"]) for r in rows})

    tag = register_source(
        kind="summary",
        title=SOURCE_TITLES[source],
        source=source,
        location="pages " + ", ".join(map(str, pages)),
        text=summary,
    )

    return f"[{tag}] Summary of {SOURCE_TITLES[source]} about '{focus}':\n{summary}"


# ==============================
# Tools Schema
# ==============================

TOOL_FUNCS = {
    "search_documents": search_documents,
    "analyze_table": analyze_table,
    "calculate": calculate,
    "summarize_document": summarize_document,
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Search the report texts (World Bank Egypt Economic Monitor Dec 2021 and Dec 2022, IMF Country Report July 2025) for explanations, causes, policies and recommendations. Returns passages with source tags.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "publisher": {"type": "string", "enum": ["World Bank", "IMF"]},
                    "source": {"type": "string", "enum": list(SOURCE_TITLES)},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_table",
            "description": "Compute numbers from World Bank indicator tables (all countries, years up to 2025). Never answer numeric questions from memory. Always pass start_year and end_year when a period is named. Operation rank orders countries by the value in the latest year of the range; rank_average orders by the average over the range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "indicator": {"type": "string", "description": "GDP growth, inflation, unemployment, GDP, or population"},
                    "countries": {"type": "array", "items": {"type": "string"}},
                    "start_year": {"type": "integer"},
                    "end_year": {"type": "integer"},
                    "operation": {"type": "string", "enum": OPERATIONS},
                },
                "required": ["indicator", "countries", "operation"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Perform arithmetic ONLY using numerical values already returned "
                "by analyze_table or search_documents in earlier tool results. "
                "NEVER invent, guess, recall, or assume input numbers. "
                "You MUST retrieve the required values from the appropriate source "
                "BEFORE calling this tool. "
                "For changes between years in a World Bank indicator, prefer "
                "analyze_table with operation='change' or operation='pct_change' "
                "instead of this calculator. "
                "Use this tool only when additional arithmetic is necessary."
            ),
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_document",
            "description": "Summarize what one report says about a focus topic. Use only when an overview of a whole document is requested.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "enum": list(SOURCE_TITLES)},
                    "focus": {"type": "string"},
                },
                "required": ["source", "focus"],
            },
        },
    },
]


# ==============================
# System Prompt and Planner
# ==============================

SYSTEM_PROMPT = """You are a research assistant on Egypt's economy. Use ONLY the tools. Cite every fact with its tag like [S1]; never invent tags.
Data: World Bank Egypt Economic Monitor (Dec 2021, Dec 2022), IMF Country Report 25/186 (Jul 2025); World Bank indicator tables (GDP growth, inflation, unemployment, GDP, population; all countries; up to 2025).
Numeric reasoning:
- NEVER invent or assume numerical values.
- Retrieve all required numerical inputs from tools before performing calculations.
- For World Bank indicator comparisons across years, use analyze_table with:
  - change for absolute differences.
  - pct_change for relative percentage changes.
  - average for averages.
  - cagr for compound annual growth.
- Do NOT call calculate before retrieving the required numerical inputs.
- Use calculate only for additional arithmetic that cannot be performed directly by analyze_table.
- Every numerical conclusion must be traceable to retrieved evidence.
- Distinguish percentage-point changes from percentage changes.
- When comparing two inflation rates or other percentages:
  - Use "percentage points" for the absolute difference.
  - Use "percent increase/decrease" for the relative change.
  - Never confuse the two.
  - Use the exact calculation returned by analyze_table.
Coverage: the World Bank reports end in 2022; the IMF report covers up to early 2025. If asked about a later period, say it is not covered by that source. IMF figures use fiscal years (FY2023/24 = Jul 2023 to Jun 2024); World Bank 'GDP growth 2024' appears to equal FY2023/24, so state the period you mean.
Before saying that something is missing from the documents, search at least twice with different wording (use the exact terms, numbers and units a report would use). Search queries must contain only content words about the topic: never include the report title, publisher or date (the filters already restrict the source). If two searches fail, change the topic words (synonyms, section names such as outlook or projections, specific figures) instead of rephrasing the same words. For rankings in a given year use operation rank (not average); use rank_average only for averages. If the tools lack the information, say what is missing. Off-topic questions: say they are outside the knowledge base. Output only the final answer, with no reasoning or notes to yourself. Maximum 150 words."""


def make_plan(question):
    planner_prompt = f"""
You are the planning component of an Egypt Economy Analyst Agent.

AVAILABLE TOOLS:
1. search_documents:
   Search World Bank and IMF reports for economic explanations,
   causes, policies, recommendations, forecasts, and context.

2. analyze_table:
   Query structured economic indicators:
   GDP growth, inflation, unemployment, GDP, and population.
   Supports values, average, rank, rank_average,
   change, pct_change, min, max, latest, and cagr.

3. calculate:
   Perform arithmetic using numbers already retrieved from tools.
   Use only when analyze_table cannot perform the calculation.

4. summarize_document:
   Summarize an entire report around a focus topic.
   Use only when the user requests a report summary.

PROJECT SCOPE:
The project focuses on Egypt's economy and its comparison
with other countries using the available indicators and reports.

AVAILABLE DATA:
- Economic indicators: all countries, years up to 2025.
- World Bank reports: December 2021 and December 2022.
- IMF Country Report: July 2025.

PLANNING RULES:

1. OUT OF SCOPE:
Return exactly "Out of scope" ONLY when the question is
unrelated to economics, Egypt's economy, or the available
economic indicators and reports.
Examples:
- "Write me a poem" -> Out of scope.
- "Explain Python loops" -> Out of scope.
- "What caused inflation in Egypt?" -> IN SCOPE.
- "Compare Egypt and Morocco's GDP growth" -> IN SCOPE.
- "What did the IMF recommend for Egypt?" -> IN SCOPE.

2. MISSING INFORMATION:
Do NOT classify a question as Out of scope just because:
- The requested year is beyond report coverage.
- A particular indicator is unavailable.
- The requested information may not exist in the documents.

Instead, plan the relevant search or data query.
The agent will determine whether the information is available.

3. TOOL SELECTION:

- Questions asking for numerical values of available indicators
  (GDP growth, inflation, unemployment, GDP, population)
  must use analyze_table.
- This rule applies regardless of the requested year.
  If the requested year is unavailable, analyze_table should
  report that the data is unavailable.
- Questions asking about causes, explanations, economic policies,
  recommendations, or historical context must use search_documents.
- Questions asking about forecasts or future projections should
  search_documents for relevant forecasts and projections.
  Do not treat actual historical data as a forecast.
- Questions requiring both numerical data and explanations
  must use both analyze_table and search_documents.
- Use calculate only when an additional mathematical calculation
  is needed based on retrieved data.
- Use summarize_document only when the user requests a report summary.
- For numerical indicator questions about future years,
  use BOTH analyze_table and search_documents:
  analyze_table checks actual data availability,
  while search_documents checks for forecasts or projections.
- Never assume that a future-year value is available
  just because a report contains other economic projections.

4. EFFICIENT SEARCH:

- Minimize tool calls.
- Combine related topics into one search query whenever possible.
- If a question asks about multiple indicators over the same period,
  retrieve the indicators separately using analyze_table,
  but search the reports only once for their shared economic context.
- Do not make separate searches for GDP growth and unemployment
  if one search can cover both.
- Do not repeat a search with nearly identical wording.
- Search a different topic only when the first search does not
  provide relevant evidence for that topic.
- Prefer one focused search per relevant report.
- Search another report only when needed for coverage or corroboration.

5. COMPARISONS:

- Compare countries using the same indicator and period.
- Use the same year for rankings whenever possible.
- Distinguish calendar years from fiscal years.
- Do not treat fiscal-year values as calendar-year values.
- Do not attribute a specific numerical change to a cause unless
  the retrieved source explicitly supports that connection.

6. SOURCE COVERAGE:

- World Bank reports end in 2022.
- IMF Country Report July 2025 covers later developments.
- If a question asks about events beyond a source's coverage,
  do not attribute those events to that source.
- Explain whether a value represents a calendar year or fiscal year
  whenever the distinction matters.

7. MINIMAL PLAN:

Return 1-3 short numbered steps.
Mention the tool and what it should retrieve.
Do not answer the question.
Do not invent data.
Avoid redundant tool calls.

Return only the plan, or exactly "Out of scope".

Question: {question}
"""

    return llm_text(planner_prompt, model=HELPER_MODEL)


# ==============================
# Guards: Numbers, Quotes, Coverage, Absence
# ==============================

NUM = re.compile(
    r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)(\s*(?:million|billion|trillion|bn)\b)?",
    re.I,
)

PCT_AFTER = re.compile(r"\s*(?:%|percent|per cent|percentage)", re.I)


def numbers_in(text, skip_scaled=False):
    out = []

    for m in NUM.finditer(str(text)):
        if skip_scaled and m.group(2):
            continue

        try:
            value = float(m.group(1).replace(",", ""))
            out.append((m.group(1), value))
        except ValueError:
            pass

    return out


def check_numbers(answer, evidence, steps=None):
    ev = [value for _, value in numbers_in(evidence)]

    if steps:
        for step in steps:
            if step.get("tool") == "calculate":
                ev.extend(
                    value for _, value in numbers_in(str(step.get("result", "")))
                )

    clean = re.sub(r"\[S\d+\]", " ", answer)
    bad = []

    for m in NUM.finditer(clean):
        if m.group(2):
            continue

        raw = m.group(1)

        try:
            val = float(raw.replace(",", ""))
        except ValueError:
            continue

        is_pct = bool(PCT_AFTER.match(clean[m.end():m.end() + 14]))
        dec = len(raw.split(".")[1]) if "." in raw else 0

        if dec == 0 and not is_pct and (val < 1000 or 1900 <= val <= 2100):
            continue

        tol = 0.5 * 10 ** (-dec) if dec else 0.5

        if not any(abs(val - e) <= tol for e in ev):
            bad.append(raw)

    return [
        f"The number {n} is not in the tool results and was not computed with calculate"
        for n in dict.fromkeys(bad)
    ]


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", s.lower())).strip()


QUOTE_PATTERN = re.compile(r'[“"]([^”"]{15,})[”"]')


def check_quotes(answer, evidence_norm):
    problems = []

    for q in QUOTE_PATTERN.findall(answer):
        for seg in re.split(r"…|\.{3}", q):
            seg_n = norm(seg)

            if len(seg_n.split()) < 4:
                continue

            if seg_n not in evidence_norm:
                problems.append(
                    f"The quoted text '{seg.strip()[:70]}' "
                    "does not appear verbatim in the sources"
                )

    return problems


def gather_evidence(steps, cited=None):
    parts = []

    for s in steps:
        if s["tool"] != "search_documents":
            parts.append(f"[{s['tool']}] {s['result']}")

    for tag, s in SOURCES.items():
        if s.get("text") and (cited is None or tag in cited):
            source_info = (
                f"[{tag}] "
                f"Title: {s.get('title', 'Unknown source')} | "
                f"Publisher: {s.get('publisher', 'Unknown')} | "
                f"Location: {s.get('location', 'Unknown')}"
            )
            parts.append(source_info + "\n" + s["text"])

    return "\n\n".join(parts)


def similar_search(args, previous):
    a = set(tok(args.get("query", "")))

    for p in previous:
        if (
            p.get("source") == args.get("source")
            and p.get("publisher") == args.get("publisher")
        ):
            b = set(tok(p.get("query", "")))

            if a and b and len(a & b) / len(a | b) >= 0.5:
                return True

    return False


COVERS_UNTIL = {"World Bank": 2022, "IMF": 2025}

REPORT_WORDS = re.compile(
    r"\b(say|says|said|report|reports|monitor|describe|describes|state|states|stated|note|notes|noted)\b",
    re.I,
)

TABLE_WORDS = re.compile(r"\b(indicator|indicators|table|wdi|data|dataset)\b", re.I)


def coverage_conflict(question):
    q = question.lower()

    if (
        "world bank" not in q
        or "imf" in q
        or TABLE_WORDS.search(q)
        or not REPORT_WORDS.search(q)
    ):
        return []

    return sorted({
        int(y)
        for y in re.findall(r"\b(20[12]\d)\b", question)
        if int(y) > COVERS_UNTIL["World Bank"]
    })


def coverage_notice(years):
    ys = ", ".join(map(str, years))
    return (
        "Note: the World Bank reports in this knowledge base "
        "(December 2021 and December 2022) end in 2022, "
        f"so they cannot say what happened in {ys}."
    )


def check_coverage(question, answer):
    if coverage_conflict(question) and re.search(r"world bank", answer, re.I):
        return [
            "Do not attribute anything about the later year to the World Bank. "
            "Use the IMF report and label it as the IMF's view, or say it is not covered."
        ]

    return []


ABSENT = re.compile(
    r"does not (?:contain|give|provide|mention|include|report|state|discuss|specify)|"
    r"do not (?:contain|provide)|"
    r"not (?:mentioned|provided|available|stated|specified)|"
    r"(?:couldn't|could not) find|"
    r"no (?:explicit|specific) ",
    re.I,
)

SCOPE_WORDS = re.compile(
    r"\b(egypt|egyptian|pound|egp|inflation|cpi|gdp|growth|"
    r"unemployment|jobs|population|econom\w*|imf|world bank|"
    r"central bank|cbe|exchange rate|currency|fiscal|monetary|"
    r"debt|reserves|interest rate|policy rate|export\w*|"
    r"import\w*|suez|subsid\w*|poverty|budget|deficit|wages?)\b",
    re.I,
)


def in_scope_hint(question):
    return bool(SCOPE_WORDS.search(question))


OUT_OF_SCOPE_ANSWER = (
    "This question is outside the knowledge base, which covers Egypt's economy: "
    "the World Bank and IMF reports and the World Bank indicator tables."
)


def normalize_citations(text):
    text = re.sub(r"[【\[]\s*(S\d+)\s*(?:†[^】\]]*)?[】\]]", r"[\1]", text)
    text = re.sub(r"【[^】]*】", "", text)

    # Drop leaked reasoning that precedes the real answer
    if "## Answer" in text:
        text = text.split("## Answer", 1)[1].strip()

    return text


# ==============================
# Verifier
# ==============================

VERIFY_PROMPT = """You are a strict evidence-based fact checker.
Your job is to verify the ANSWER using ONLY the provided EVIDENCE.

Question:
<<Q>>

EVIDENCE:
<<E>>

ANSWER:
<<A>>

Follow these rules carefully:

1. Check every factual claim separately.
2. Every numerical value must appear in the evidence or be correctly calculated from supported values.
3. Check calculations carefully. Distinguish percentage points from relative percentage changes.
4. Check the time period of every claim.
   - Evidence from 2022 cannot prove an event happened in 2023.
   - Evidence from 2024 or 2025 cannot automatically explain events in 2023.
5. Distinguish annual average inflation from monthly year-over-year inflation.
6. A source published in a later year may describe earlier events, but verify that the passage explicitly refers to those events.
7. A citation must exist in the evidence and directly support the sentence where it appears.
8. Do not accept a claim just because it sounds economically reasonable.
9. If the evidence is insufficient, mark the answer as not fully supported.
10. Do not confuse correlation, context, or policy responses with proven causes.
11. Do not interpret missing evidence as proof that something
does not exist or did not happen.

For example, if the knowledge base contains no World Bank
report covering 2023, do not claim that the World Bank
never identified the causes of inflation in 2023.
State only that the available evidence does not establish this.

Return ONLY valid JSON in exactly this format:

{ "ok": true,
  "problems": [] }

If ok is true, problems must be an empty list.
If ok is false, problems must be a list of specific explanations.
Do not return problems as a string.

Set ok=false if ANY important factual claim is unsupported.
Set ok=true only if all important claims are supported."""


def verify_answer(question, answer, evidence):
    try:
        raw = llm_text(
            VERIFY_PROMPT.replace("<<Q>>", question)
            .replace("<<E>>", evidence)
            .replace("<<A>>", answer),
            model=HELPER_MODEL,
        )
    except RuntimeError:
        raise
    except Exception:
        return None, ["verifier unavailable"]

    try:
        start = raw.find("{")
        if start == -1:
            return None, ["verifier output could not be parsed"]

        data, _ = json.JSONDecoder().raw_decode(raw[start:])
        ok = data.get("ok")
        problems = data.get("problems")

        if not isinstance(ok, bool):
            return None, ["invalid verifier status"]

        if isinstance(problems, str):
            text = problems.strip().lower()
            problems = [] if text in ["", "none", "no problems", "no issues"] else [problems]

        if not isinstance(problems, list):
            return None, ["invalid verifier problems format"]

        problems = [str(p).strip() for p in problems[:6] if str(p).strip()]

        if ok and problems:
            return False, problems

        if not ok and not problems:
            return False, ["Answer was rejected without a specific explanation"]

        return ok, problems

    except Exception:
        return None, ["verifier output could not be parsed"]


def revise_answer(question, answer, problems, evidence):
    prompt = f"""You are a strict evidence-based answer editor.

Rewrite the answer using ONLY the provided evidence.

Question:
{question}

Evidence:
{evidence}

Previous answer:
{answer}

Problems found:
{problems}

Rules:
- Remove unsupported claims.
- Never invent missing facts, numbers, or causes.
- Preserve correct numerical values.
- Distinguish percentage points from percentage changes.
- Clearly distinguish calendar years from fiscal years.
- Keep citations such as [S1].
- Every factual claim must have supporting evidence.
- Do not treat missing evidence as proof that something did not happen.
- If evidence is insufficient, say:
  "The available sources do not establish this."
- Do not claim that an organization never published or stated something
  unless the evidence explicitly supports that claim.
- Maximum 150 words.

Return ONLY the corrected answer."""

    return normalize_citations(llm_text(prompt, model=HELPER_MODEL))


# ==============================
# Agent Loop
# ==============================

def agent_loop(messages, steps, seen, searches, max_steps):
    tool_count = 0

    for _ in range(max_steps + 2):

        try:
            msg = llm_chat(messages, tools=TOOLS)

        except Exception as e:
            if getattr(e, "status_code", None) == 400:
                messages.append({
                    "role": "user",
                    "content": "The last tool call was malformed. Call the tool again with valid JSON arguments.",
                })
                continue
            raise

        if not msg.tool_calls:
            return (msg.content or "").strip()

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {
                        "name": c.function.name,
                        "arguments": c.function.arguments,
                    },
                }
                for c in msg.tool_calls
            ],
        })

        budget_reached = False

        for c in msg.tool_calls:
            args = {}

            if tool_count >= max_steps:
                result = "Tool-call limit reached. Use the evidence already collected and answer now."
                budget_reached = True

            else:
                try:
                    args = json.loads(c.function.arguments or "{}")
                    tool_count += 1

                    key = (c.function.name, json.dumps(args, sort_keys=True))

                    if key in seen:
                        result = f"Duplicate call (same result as step {seen[key]}). Do not repeat it."

                    elif c.function.name == "search_documents" and similar_search(args, searches):
                        result = "A very similar search was already done. Use different wording or another source."

                    else:
                        result = TOOL_FUNCS[c.function.name](**args)
                        seen[key] = len(steps) + 1

                        if c.function.name == "search_documents":
                            searches.append(args)

                except Exception as e:
                    result = f"Tool error: {type(e).__name__}: {e}"

            steps.append({"tool": c.function.name, "args": args, "result": result})

            messages.append({
                "role": "tool",
                "tool_call_id": c.id,
                "content": str(result)[:2200],
            })

        if budget_reached or tool_count >= max_steps:
            messages.append({
                "role": "user",
                "content": (
                    "Tool-call limit reached. Stop calling tools and "
                    "answer using only the evidence already collected. "
                    "If evidence is insufficient, explain what is missing."
                ),
            })
            return (llm_chat(messages).content or "").strip()

    messages.append({
        "role": "user",
        "content": (
            "Step limit reached. Answer now using only the information "
            "already gathered. If it is not enough, say what is missing."
        ),
    })

    return (llm_chat(messages).content or "").strip()


def confidence(verified, guard_rounds, revised, steps, late):
    if not steps:
        return {"score": None, "label": "n/a"}

    score = 100
    text_used = any(s["tool"] in ("search_documents", "summarize_document") for s in steps)

    if text_used and verified is None:
        score -= 25
    if verified is False:
        score -= 30
    if revised:
        score -= 10

    score -= 10 * guard_rounds

    if late:
        score -= 10

    if any(
        str(s["result"]).startswith(("Tool error", "No relevant", "Unknown", "No data"))
        for s in steps
    ):
        score -= 10

    score = max(score, 0)

    return {
        "score": score,
        "label": "High" if score >= 80 else "Medium" if score >= 50 else "Low",
    }


def run_agent(question, max_steps=6, verify=True):
    global CURRENT_QUESTION
    CURRENT_QUESTION = question
    SOURCES.clear()
    LLM_STATS.update(calls=0, tokens=0)
    LLM_STATS["log"].clear()
    t0 = time.time()

    def result(answer, plan="", steps=(), verified=None, problems=(), revised=False,
               guard_rounds=0, guard_problems=(), late=()):
        cited = sorted(
            {t for t in re.findall(r"\bS\d+\b", answer) if t in SOURCES},
            key=lambda t: int(t[1:]),
        )
        return {
            "question": question,
            "plan": plan,
            "steps": list(steps),
            "answer": answer,
            "sources": {t: SOURCES[t] for t in cited},
            "guard_rounds": guard_rounds,
            "guard_problems": list(guard_problems),
            "verified": verified,
            "problems": list(problems),
            "revised": revised,
            "coverage_years": list(late),
            "confidence": confidence(verified, guard_rounds, revised, steps, late),
            "llm_calls": LLM_STATS["calls"],
            "tokens": LLM_STATS["tokens"],
            "seconds": round(time.time() - t0, 1),
        }

    late = coverage_conflict(question)

    if not in_scope_hint(question):
        return result(OUT_OF_SCOPE_ANSWER, "Out of scope (no economy keywords in the question)")

    try:
        plan = make_plan(question)
    except RuntimeError:
        raise
    except Exception:
        plan = "(planner unavailable)"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": "Plan for this question:\n" + plan},
    ]

    if late:
        messages.append({
            "role": "system",
            "content": (
                f"The question concerns {late}, but the World Bank reports "
                "in this knowledge base end in 2022. Do not attribute "
                "2023 events or explanations to these reports. "
                "You may explain that the requested information is not "
                "available in the World Bank reports provided. "
                "Do not substitute IMF findings for World Bank statements. "
                "Clearly label any IMF evidence separately."
            ),
        })

    messages.append({"role": "user", "content": question})

    steps, seen, searches = [], {}, []
    final = normalize_citations(agent_loop(messages, steps, seen, searches, max_steps))
    guard_rounds, guard_problems = 0, []

    if steps and not late and ABSENT.search(final) and len(searches) < 3:
        guard_rounds += 1
        guard_problems.append("absence claim: asked to search again")
        messages.append({"role": "assistant", "content": final})
        messages.append({
            "role": "user",
            "content": (
                "Before concluding that the information is missing, search again with clearly "
                "different topic words (synonyms, section names, specific figures; no report title or date). "
                "Then give the final answer."
            ),
        })
        final = normalize_citations(agent_loop(messages, steps, seen, searches, 3))

    for _ in range(2):
        if not steps:
            break

        evidence = gather_evidence(steps)

        found = (
            check_numbers(final, evidence, steps)
            + check_quotes(final, norm(evidence))
            + check_coverage(question, final)
        )

        if not found:
            break

        guard_rounds += 1
        guard_problems += found

        messages.append({"role": "assistant", "content": final})
        messages.append({
            "role": "user",
            "content": (
                "Automatic check found problems: " + "; ".join(found)
                + ". Fix them: compute figures with calculate "
                "(or remove them), paraphrase instead of quoting, "
                "and follow the coverage rule. Then give the corrected final answer."
            ),
        })

        final = normalize_citations(agent_loop(messages, steps, seen, searches, 3))

    verified, problems, revised = None, [], False

    used_text = any(s["tool"] in ("search_documents", "summarize_document") for s in steps)

    if verify and used_text:
        cited_now = {t for t in re.findall(r"\bS\d+\b", final) if t in SOURCES}
        evidence = gather_evidence(steps, cited=cited_now)[:6000]

        verified, problems = verify_answer(question, final, evidence)

        if verified is False:
            final = normalize_citations(revise_answer(question, final, problems, evidence))
            revised = True

            cited_now = {t for t in re.findall(r"\bS\d+\b", final) if t in SOURCES}
            evidence = gather_evidence(steps, cited=cited_now)[:6000]

            verified, problems = verify_answer(question, final, evidence)

            if verified is not True:
                final += (
                    "\n\nVerification warning: "
                    "This answer could not be fully verified "
                    "against the available evidence."
                )

    if late:
        final = coverage_notice(late) + "\n\n" + final

    return result(final, plan, steps, verified, problems, revised,
                  guard_rounds, guard_problems, late)


# ==============================
# Display Helper
# ==============================

def show_run(res):
    print("PLAN:", res["plan"].replace("\n", " | "))

    for i, s in enumerate(res["steps"], 1):
        print(f"  step {i}: {s['tool']}({json.dumps(s['args'], ensure_ascii=False)})")
        print("      ->", str(s["result"])[:120].replace("\n", " "))

    print("\nANSWER:", res["answer"])

    for t, s in res["sources"].items():
        print(f"  [{t}] {s['title']} | {s['location']}")

    status = {True: "passed", False: "FAILED -> revised", None: "not verified"}[res["verified"]]
    print(f"\nGuards: {res['guard_rounds']} round(s) {res['guard_problems']}")

    if res.get("coverage_years"):
        print("Coverage notice added for:", res["coverage_years"])

    print(f"Verifier: {status} {res['problems'] if res['problems'] else ''}")

    c = res.get("confidence")
    if c:
        print(f"Confidence: {c['label']} ({c['score']})")

    if res.get("answer_en") and res.get("language") == "ar":
        print("\nEnglish answer:", res["answer_en"])

    print(f"[{res['llm_calls']} LLM calls | {res['tokens']} tokens | {res['seconds']}s]")


# ==============================
# Arabic Translation Layer
# ==============================

def detect_language(text):
    return "ar" if re.search(r"[\u0600-\u06FF]", text) else "en"


def translate_to_english(text):
    prompt = f"""
Translate the following question from Arabic to English.

Rules:
- Preserve the exact meaning.
- Keep numbers, dates, and economic terms accurate.
- Do not answer the question.
- Return translation only.

Question:
{text}
"""
    return llm_text(prompt, model=HELPER_MODEL)


def translate_to_arabic(text):
    prompt = f"""
Translate the following answer into clear, natural Arabic.

Rules:
- Preserve all facts, numbers, percentages, and dates.
- Use Western digits (0-9), not Arabic-Indic digits.
- Use the % sign, not the Arabic percent sign.
- Keep citations like [S1], [S2] unchanged.
- Keep organization names such as World Bank and IMF recognizable.
- Do not add or remove information.
- Use natural Arabic suitable for economic analysis.
- Return only the translated answer.
- Use these Arabic terms: exchange rate = سعر الصرف; flexible exchange rate = سعر صرف مرن; foreign exchange = النقد الأجنبي; liberalized = محرر; Central Bank of Egypt = البنك المركزي المصري; Egyptian pound = الجنيه المصري; inflation = التضخم.
- Check the spelling of every Arabic word before answering.

Answer:
{text}
"""
    return llm_text(prompt, model=HELPER_MODEL)


def run_multilingual_agent(question, max_steps=6, verify=True):
    t_start = time.time()
    lang = detect_language(question)
    extra_calls = extra_tokens = 0
    q_en = question

    if lang == "ar":
        c0, t0 = LLM_STATS["calls"], LLM_STATS["tokens"]
        q_en = translate_to_english(question)
        extra_calls += LLM_STATS["calls"] - c0
        extra_tokens += LLM_STATS["tokens"] - t0

    res = run_agent(q_en, max_steps=max_steps, verify=verify)
    res.update(language=lang, question_en=q_en, answer_en=res["answer"])

    if lang == "ar":
        c0, t0 = LLM_STATS["calls"], LLM_STATS["tokens"]
        res["answer"] = translate_to_arabic(res["answer_en"])
        extra_calls += LLM_STATS["calls"] - c0
        extra_tokens += LLM_STATS["tokens"] - t0

    res["llm_calls"] += extra_calls
    res["tokens"] += extra_tokens
    res["model"] = MODEL
    res["seconds"] = round(time.time() - t_start, 1)
    return res


# ==============================
# Evaluation (run once, resume from CSV)
# ==============================

EVAL_SET = [
    dict(id="N1", cat="numeric", q="What was Egypt's inflation in 2023 compared with 2020, and how many times higher was it?",
         expected="33.88% vs 5.04%, about 6.7 times", keys=[r"33\.88", r"5\.04", r"6\.7"]),
    dict(id="N2", cat="numeric", q="Which of Egypt and Jordan had the higher average unemployment between 2015 and 2022, and by how many percentage points?",
         expected="Jordan 17.35 vs Egypt 9.62, gap 7.73", keys=[r"Jordan", r"7\.7"]),
    dict(id="N3", cat="numeric", q="By what percentage did Egypt's population grow between 2015 and 2024?",
         expected="about 17.0% (99.60M to 116.54M)", keys=[r"17\.0"]),
    dict(id="N4", cat="numeric", q="What was Egypt's compound annual growth rate of GDP in current US dollars between 2019 and 2024?",
         expected="about 4.07% per year", keys=[r"4\.07"]),
    dict(id="N5", cat="numeric", q="Rank Egypt, Turkey, Morocco and Jordan by population in 2024.",
         expected="Egypt, Turkiye, Morocco, Jordan, using the 2024 values", keys=[r"Egypt.*Turk.*Morocco.*Jordan", r"116"]),
    dict(id="T1", cat="text", q="According to the World Bank's December 2022 Egypt Economic Monitor, what caused inflation to accelerate in 2022?",
         expected="global food and energy prices, exchange rate depreciation, supply bottlenecks", keys=[r"food", r"energy", r"deprec", r"bottleneck"]),
    dict(id="T2", cat="text", q="By how many basis points did the Central Bank of Egypt cumulatively raise policy rates in 2022, according to the World Bank?",
         expected="500 basis points (March, May, October 2022)", keys=[r"500"]),
    dict(id="T3", cat="text", q="How much had the Egyptian pound depreciated cumulatively by the end of October 2022, according to the World Bank?",
         expected="53.5% (to about LE24 per US$)", keys=[r"53\.5"]),
    dict(id="T4", cat="text", q="How much access does the arrangement under the Resilience and Sustainability Facility provide, according to the IMF report?",
         expected="about US$1.3 billion (SDR 1 billion)", keys=[r"1\.3|SDR 1 billion"]),
    dict(id="T5", cat="text", q="What was Egypt's real GDP growth in FY2023/24 according to the IMF, and how does it compare with the previous year?",
         expected="2.4% vs 3.8%", keys=[r"2\.4", r"3\.8"]),
    dict(id="M1", cat="multi", q="How does the IMF describe the trend of Egypt's inflation in 2024, and how does that differ from the World Bank's description of 2022?",
         expected="IMF: declining in 2024; World Bank: accelerating to double digits in 2022 (judge manually)", keys=[r"declin|eas|downward|fell|lower", r"acceler|rising|surge|double"]),
    dict(id="M2", cat="multi", q="What was Egypt's GDP growth in 2022 according to the indicator table, and what does the World Bank's December 2022 report say about the economic outlook?",
         expected="6.59% from the table + outlook from the report (judge manually)", keys=[r"6\.59|6\.6"]),
    dict(id="R1", cat="refusal", q="What did the World Bank say caused Egypt's inflation in 2023?",
         expected="should say the World Bank reports end in 2022 and not attribute 2023 to them", keys=[r"end in 2022|cannot say|not cover"]),
    dict(id="R2", cat="refusal", q="Who won the 2018 FIFA World Cup?",
         expected="should say it is outside the knowledge base", keys=[r"outside"]),
]


def auto_check(item, answer):
    return all(re.search(k, answer, re.I | re.S) for k in item["keys"])


EVAL_CSV = BASE_DIR / "eval_results.csv"
RESULTS = []


def run_eval(ids=None, pause=20):
    if not RESULTS and EVAL_CSV.exists():
        RESULTS.extend(pd.read_csv(EVAL_CSV).to_dict("records"))

    RESULTS[:] = [r for r in RESULTS if not str(r["answer"]).startswith("ERROR")]

    for item in EVAL_SET:
        if ids and item["id"] not in ids:
            continue

        if any(r["id"] == item["id"] for r in RESULTS):
            continue

        try:
            out = run_agent(item["q"])

        except RuntimeError as e:
            print("\nSTOPPED:", str(e)[:200])
            return

        except Exception as e:
            out = {
                "answer": f"ERROR {type(e).__name__}: {str(e)[:100]}",
                "llm_calls": 0, "tokens": 0, "seconds": 0,
                "verified": None, "guard_rounds": 0, "revised": False,
            }

        RESULTS.append({
            "id": item["id"],
            "cat": item["cat"],
            "question": item["q"],
            "expected": item["expected"],
            "answer": out["answer"],
            "auto_ok": auto_check(item, out["answer"]),
            "verified": out["verified"],
            "guard_rounds": out["guard_rounds"],
            "revised": out["revised"],
            "llm_calls": out["llm_calls"],
            "tokens": out["tokens"],
            "seconds": out["seconds"],
            "model": MODEL,
        })

        pd.DataFrame(RESULTS).to_csv(EVAL_CSV, index=False, encoding="utf-8-sig")

        r = RESULTS[-1]
        total = sum(x["tokens"] for x in RESULTS)

        print(
            f"{r['id']:3} auto={str(r['auto_ok']):5} "
            f"verified={str(r['verified']):5} "
            f"guards={r['guard_rounds']} "
            f"calls={r['llm_calls']:2} "
            f"tokens={r['tokens']:6} (total {total}) "
            f"{r['seconds']:5}s"
        )

        time.sleep(pause)


def show_results():
    for r in RESULTS:
        print(
            f"\n[{r['id']}] expected: {r['expected']}\n"
            f"   -> {str(r['answer'])[:450].replace(chr(10), ' ')}"
        )