#!/usr/bin/env python3
"""
ingest_lc214.py
-------------------------------------------------------------------------------
Scrapes Lei Complementar 214/2025 directly from the Diário Oficial portal,
**splits it into article-level text chunks (~350 tokens)**, extracts every
<table> as CSV/Markdown, and writes everything to ./output/ for your RAG stack.

Pedagogical comments explain each step so you can modify:

*   how the law is downloaded & cached           (see `download_html`)
*   how tables are cleaned & normalised          (see `extract_tables`)
*   how article hierarchies are detected         (see `split_articles`)
*   how child-chunks are produced                (adjust `CHUNK_SIZE`)
*   where metadata is injected for retrieval     (edit the `meta` dict)

Run `pip install requests beautifulsoup4 lxml pandas langchain tiktoken`
once, then execute the script.  Re-runs are 🚀 fast thanks to ETag caching.

Quick‑start
-----------
1.  pip install -r requirements.txt
2.  python ingest_lc214.py --url "<DOU‑URL>" --out ./output
3.  Inspect output/ for articles.jsonl and tables/
   • law214.md for a complete markdown snapshot
"""
from __future__ import annotations

# ── standard libs ────────────────────────────────────────────────────────────
import argparse
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
import io

# ── third-party libs (all pure-Python except pandas) ─────────────────────────
import requests                      # network fetch w/ HTTP caching headers
import pandas as pd                  # table extraction + CSV/MD output
from bs4 import BeautifulSoup        # robust HTML parsing
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter

# ──---------------------------------------------------------------------------
#  CONFIGURABLE CONSTANTS
# ──---------------------------------------------------------------------------

# Change User-Agent if you want to mimic a different client or avoid blocks
HEADERS = {                          # polite UA avoids anti-bot blocks
    "User-Agent": "lc214-scraper/1.0 (+https://github.com/your-org)"
}

# Increase if you need larger chunks for models with 32k context windows
CHUNK_SIZE    = 350                  # ≈ tokens per child-chunk

# Increase overlap if you want smoother retrieval at chunk boundaries
CHUNK_OVERLAP = 40                   # token overlap → smooth retrieval

# Increase timeout if your network is slow or server is slow to respond
REQUEST_TIMEOUT = 20                 # seconds

# Change if you want to store cache in a different subfolder
CACHE_SUBDIR    = "__cache__"        # keeps the raw HTML & headers

# ---------------------------------------------------------------------------
# 1) NETWORK LAYER -----------------------------------------------------------
# ---------------------------------------------------------------------------

def cache_paths(out_dir: Path) -> tuple[Path, Path]:
    """
    Helper: return the paths for the cached HTML file & its metadata JSON.

    out_dir / "__cache__/page.html"
    out_dir / "__cache__/meta.json"
    """
    cdir = out_dir / CACHE_SUBDIR
    cdir.mkdir(parents=True, exist_ok=True)
    return cdir / "page.html", cdir / "meta.json"


def download_html(url: str, out_dir: Path) -> str:
    """
    Download the law’s HTML **unless** the cached copy is still valid.

    Uses ETag + Last-Modified headers so Diário Oficial can reply 304.
    """
    html_path, meta_path = cache_paths(out_dir)
    headers = HEADERS.copy()         # base headers

    # ── add conditional headers (If-None-Match / If-Modified-Since) ─────────
    meta: Dict[str, str] = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("etag"):
            headers["If-None-Match"] = meta["etag"]
        if meta.get("last_modified"):
            headers["If-Modified-Since"] = meta["last_modified"]

    resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)

    # If server says “Not Modified” and we have the HTML, short-circuit
    if resp.status_code == 304 and html_path.exists():
        print("⏩  Cached HTML is up-to-date.")
        return html_path.read_text(encoding="utf-8")

    # Otherwise save the fresh HTML + response headers
    resp.raise_for_status()
    html_path.write_text(resp.text, encoding="utf-8")
    meta.update(
        {
            "url": url,
            # Use timezone‑aware UTC datetime to avoid deprecation warning
            "fetched": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "etag": resp.headers.get("ETag", ""),
            "last_modified": resp.headers.get("Last-Modified", ""),
        }
    )
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print("⬇  Downloaded fresh HTML.")
    return resp.text


# ──---------------------------------------------------------------------------
#  TABLE EXTRACTION
# ──---------------------------------------------------------------------------

def clean_percent(cell: str | float | None) -> float | None:
    """
    Try to convert any substring that looks like 'NN,NN%' or 'NN.NN %' to a
    float in [0,1].  If no numeric percentage is found, return the original
    value unchanged so the caller can decide how to handle it.
    """
    import re

    if cell is None or not isinstance(cell, str):
        return cell

    # look for the *first* number followed by a '%' (e.g. '14,92537%')
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*%", cell)
    if not m:
        return cell  # let the raw string pass through unchanged

    num_txt = m.group(1).replace(".", "").replace(",", ".")
    try:
        return float(num_txt) / 100
    except ValueError:
        return cell  # fallback: unchanged


def extract_tables(soup: BeautifulSoup, out_dir: Path) -> List[Dict[str, Any]]:
    """
    Iterate over every <table> in the HTML, normalise it, and write:

    * CSV  → machine-readable for tool functions
    * MD   → nice plain-text for embedding / LLM context

    Returns:
        list of metadata dicts: {id, caption, n_rows, n_cols}
    """

    # To disable percentage normalization, comment out or remove the call to clean_percent
    # in the loop below where columns containing '%' are detected.
    # This preserves the original string values in the DataFrame.

    (out_dir / "tables").mkdir(exist_ok=True)
    metas: List[Dict[str, Any]] = []

    for idx, tbl in enumerate(soup.select("table")):
        tid = f"TBL_{idx+1:02d}"     # stable table ID (TBL_01 …)

        # ---- Determine a human-readable caption --------------------------
        caption_tag = tbl.find_previous(lambda t: t.name == "p"
                                                  and t.get_text(strip=True))
        caption = caption_tag.get_text(" ", strip=True) if caption_tag else f"Tabela {idx+1}"

        # ---- Convert HTML → DataFrame (pandas auto-handles rowspan) -----
        # Wrap HTML string in StringIO to avoid the future deprecation warning
        df: pd.DataFrame = pd.read_html(
            io.StringIO(str(tbl)), decimal=",", thousands="."
        )[0]

        # Drop pandas’ auto-generated index column when present
        if re.match(r"Unnamed.*0", str(df.columns[0])):
            df = df.drop(columns=df.columns[0])

        # Merge two-row headers (happens in DOU tables)
        if df.iloc[0].isna().sum():            # first data row is header row 2
            df.columns = [
                h2 if pd.notna(h2) else h1 for h1, h2 in zip(df.columns, df.iloc[0])
            ]
            df = df.iloc[1:].copy()

        # Clean % strings → floats (robust against descriptive text rows)
        for col in df.columns:
            if df[col].astype(str).str.contains("%").any():
                df[col] = df[col].apply(clean_percent)

        # ---- Persist to disk --------------------------------------------
        df.to_csv(out_dir / "tables" / f"{tid}.csv", index=False)
        (out_dir / "tables" / f"{tid}.md").write_text(df.to_markdown(index=False),
                                                      encoding="utf-8")

        metas.append({"id": tid,
                      "caption": caption,
                      "n_rows": int(df.shape[0]),
                      "n_cols": int(df.shape[1])})

    # Save a quick index JSON for look-ups
    (out_dir / "tables_index.json").write_text(json.dumps(metas, ensure_ascii=False, indent=2),
                                               "utf-8")
    print(f"✅  Extracted {len(metas)} tables.")
    return metas


# ──---------------------------------------------------------------------------
#  HTML → MARKDOWN with explicit headings
# ──---------------------------------------------------------------------------

def html_to_markdown_with_headers(soup: BeautifulSoup) -> str:
    """
    Convert the DOU law HTML (which uses only <p> tags) into markdown that
    carries real heading levels.  This lets MarkdownHeaderTextSplitter
    infer hierarchy automatically.

    Heading levels:
        #   LIVRO
        ##  TÍTULO
        ### CAPÍTULO
        #### SEÇÃO
        ##### SUBSEÇÃO
        ###### Artigo
    """
    RX = {
        "book":    re.compile(r"^LIVRO", re.I),
        "title":   re.compile(r"^T[ÍI]TULO", re.I),
        "chapter": re.compile(r"^CAP[ÍI]TULO", re.I),
        "section": re.compile(r"^(SEÇÃO|SECAO)", re.I),
        "subsection": re.compile(r"^SUBSEÇÃO|SUBSECAO", re.I),
        "article": re.compile(r"^Art\.?\s*\d+[A-Z]?\s*º?[\.:]?", re.I),
    }
    md_lines: List[str] = []
    for p in soup.select("p"):
        text = " ".join(p.stripped_strings)
        if not text:
            continue
        if   RX["book"].match(text):       md_lines.append(f"# {text}")
        elif RX["title"].match(text):      md_lines.append(f"## {text}")
        elif RX["chapter"].match(text):    md_lines.append(f"### {text}")
        elif RX["section"].match(text):    md_lines.append(f"#### {text}")
        elif RX["subsection"].match(text): md_lines.append(f"##### {text}")
        elif RX["article"].match(text):    md_lines.append(f"###### {text}")
        else:                              md_lines.append(text)
    return "\n\n".join(md_lines)

# ──---------------------------------------------------------------------------
#  ARTICLE SPLITTING & CHUNKING
# ──---------------------------------------------------------------------------

def split_articles(soup: BeautifulSoup,
                   tables_meta: List[Dict[str, Any]],
                   out_dir: Path,
                   chunk_size: int = CHUNK_SIZE,
                   chunk_overlap: int = CHUNK_OVERLAP):
    """
    Walk through <p> tags in order, detect hierarchy lines, and emit
    child-chunks (~350 tokens) with metadata.  The heavy lifting of token
    splitting is delegated to LangChain’s *RecursiveCharacterTextSplitter*.

    To add new hierarchy levels, extend the RX dictionary with new regex keys,
    and add corresponding metadata keys in the `meta` dictionary.
    Adjust regex patterns if the Diário Oficial formatting changes to correctly
    detect headings or articles.
    To attach extra metadata (e.g. publication date), add fields to the metadata
    dictionary and update them as needed during parsing.

    Hierarchy levels detected: book, title, chapter, section, subsection, article.
    """

    # To add new hierarchy levels, extend the RX dictionary with new regex keys,
    # and add corresponding metadata keys in the `meta` dictionary.
    # Adjust regex patterns if the Diário Oficial formatting changes to correctly
    # detect headings or articles.
    # To attach extra metadata (e.g. publication date), add fields to the metadata
    # dictionary and update them as needed during parsing.

    # ------------------------------------------------------------------
    # 0) Convert HTML soup → markdown with proper headings
    md_text = html_to_markdown_with_headers(soup)
    # Persist a full‑law markdown snapshot BEFORE further splitting
    md_file = out_dir / "law214.md"
    md_file.write_text(md_text, encoding="utf-8")
    print(f"📝  Markdown version written → {md_file.relative_to(out_dir.parent)}")

    # ------------------------------------------------------------------
    # 1) First‑level split: use headings to preserve hierarchy
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "book"),
            ("##", "title"),
            ("###", "chapter"),
            ("####", "section"),
            ("#####", "subsection"),
            ("######", "article"),
        ]
    )
    parent_docs = header_splitter.split_text(md_text)
    # ------------------------------------------------------------------
    # Persist the UNSPLIT parent documents (entire articles) so you can
    # inspect or embed them separately.
    parent_path = out_dir / "parents.jsonl"
    with parent_path.open("w", encoding="utf-8") as fp:
        for doc in parent_docs:
            fp.write(json.dumps({"text": doc.page_content,
                                 "metadata": doc.metadata},
                                ensure_ascii=False) + "\n")
    print(f"🗂️  Saved {len(parent_docs)} parent docs → {parent_path.relative_to(out_dir.parent)}")

    # ------------------------------------------------------------------
    # 2) Second‑level split: token‑size child chunks (~350 tokens)
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "."],
    )

    chunks: List[Dict[str, Any]] = []
    for parent in parent_docs:
        for child in child_splitter.split_text(parent.page_content):
            chunks.append(
                {
                    "id": str(uuid.uuid4()),
                    "text": child,
                    "metadata": parent.metadata,  # already has hierarchy
                }
            )

    # ------------------------------------------------------------------
    # 3) Persist child chunks as JSONL
    out_path = out_dir / "articles.jsonl"
    with out_path.open("w", encoding="utf-8") as fp:
        for ch in chunks:
            fp.write(json.dumps(ch, ensure_ascii=False) + "\n")
    print(f"✅  Wrote {len(chunks)} article chunks → {out_path.relative_to(out_dir.parent)}")

    return chunks


# ──---------------------------------------------------------------------------
#  MAIN ENTRY POINT
# ──---------------------------------------------------------------------------

def main():
    # ---------- CLI parsing ---------------------------------------------------
    ap = argparse.ArgumentParser(
        description="Scrape & structure LC 214/2025 for RAG ingestion."
    )
    ap.add_argument("--url", required=True, help="DOU HTML URL of the law")
    ap.add_argument("--out", type=Path, default=Path("output"),
                    help="Output folder (default: ./output)")
    ap.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    ap.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # ---------- 1. Fetch (with caching) --------------------------------------
    html = download_html(args.url, args.out)

    # ---------- 2. Parse HTML → BeautifulSoup --------------------------------
    soup = BeautifulSoup(html, "lxml")

    # ---------- 3. Extract tables --------------------------------------------
    tables_meta: List[Dict[str, Any]] = []   # manual/curated tables only

    # ---------- 4. Split prose into article chunks ---------------------------
    split_articles(soup, tables_meta, args.out,
                   chunk_size=args.chunk_size,
                   chunk_overlap=args.chunk_overlap)

    print("🏁  Done! You can now embed `articles.jsonl` + table markdowns.")

if __name__ == "__main__":
    main()

# Example:
# python ingest_lc214.py --url "https://www.in.gov.br/en/web/dou/-/lei-complementar-n-214-de-16-de-janeiro-de-2025-607430757" --chunk-size 350 --chunk-overlap 40