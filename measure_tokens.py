# measure_tokens_md.py
"""
Measure token lengths directly from the markdown version of LC-214.

Usage
-----
python measure_tokens_md.py output/law214.md
"""
import sys, statistics, collections, json, pathlib
import tiktoken
from langchain_text_splitters import MarkdownHeaderTextSplitter

if len(sys.argv) < 2:
    sys.exit("usage: python measure_tokens_md.py <path/to/law214.md>")

md_path = pathlib.Path(sys.argv[1])
md_text = md_path.read_text(encoding="utf-8")

# --- 1) Split markdown into parent docs (same headings used in scraper) -----
splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "book"), ("##", "title"), ("###", "chapter"),
        ("####", "section"), ("#####", "subsection"), ("######", "article"),
    ]
)
parents = splitter.split_text(md_text)

# --- 2) Tokenise each parent -----------------------------------------------
enc = tiktoken.encoding_for_model("text-embedding-ada-002")  # or your model
lengths = [len(enc.encode(doc.page_content)) for doc in parents]

# --- 3) Simple stats --------------------------------------------------------
print("From:", md_path)
print("parents (articles) counted:", len(lengths))
# statistics.quantiles requires keyword args; compute 95‑th percentile
q95 = statistics.quantiles(lengths, n=100)[94]   # 0‑based index
print("min / median / 95th / max:",
      min(lengths),
      statistics.median(lengths),
      round(q95, 1),
      max(lengths))

# --- 4) Histogram buckets of 50 tokens --------------------------------------
hist = collections.Counter(n//50*50 for n in lengths)
for bucket in range(0, max(hist)+50, 50):
    bar = "█" * (hist.get(bucket,0)//5)  # one block ≈5 items
    print(f"{bucket:4}-{bucket+49:>4}: {bar}")