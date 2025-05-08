#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Desenvolvido por: Ubaldino Meirelles
RAG pipeline for Lei Complementar 214/2025 (IBS + CBS + IS).

Usage examples
--------------
# (re)build the Chroma index
$ python llm_tax_rag.py --reindex

# ask a question
$ python llm_tax_rag.py --ask "tributação receita financeira"

Dependencies
------------
pip install langchain-openai langchain-community chromadb python-dotenv
"""

import argparse
import os
import shutil
import math
from pathlib import Path
from typing import List, Tuple
import numpy as np

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_community.vectorstores import Chroma
from chromadb.config import Settings
import chromadb  # new PersistentClient

from langchain.retrievers import ParentDocumentRetriever        
from langchain.storage import InMemoryStore, LocalFileStore
from langchain.prompts import PromptTemplate




# --------------------------------------------------------------------------- #
# 0.  Environment                                                             #
# --------------------------------------------------------------------------- #
load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]  

# --------------------------------------------------------------------------- #
# 1.  Constants                                                               #
# --------------------------------------------------------------------------- #
MD_PATH = "output/law214.md"
CHROMA_DIR = "docs/chroma"
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 3072  


HEADER_MAP = [
    ("#", "book"),
    ("##", "title"),
    ("###", "chapter"),
    ("####", "section"),
    ("#####", "subsection"),
    ("######", "article"),
]

PARA_SPLITTER_KWARGS = dict(
    chunk_size=1600,
    chunk_overlap=0,
    # First cut exactly at a new-article header,
    # fall back to blank lines, then single newlines
    separators=[r"\n#+\s*Art\.?", "\n\n", "\n"],
)

# --- refinamento: impedir fatias só‑cabeçalho --------------------------
# (min_length não é suportado pela API atual do TextSplitter)
PARA_SPLITTER_KWARGS.update(
    keep_separator=False,   # não duplica o “### Art. …”
)

# --- Retrieval defaults ----------------------------------------------------- #
DEFAULT_K_PARENTS  = 10   # number of parent articles to keep
DEFAULT_K_FETCH    = 1500   # child slices fetched before merge

# --------------------------------------------------------------------------- #
# 2.  Build / rebuild the vector store                                        #
# --------------------------------------------------------------------------- #
def build_vectorstore(md_path: str = MD_PATH,
                      db_path: str = CHROMA_DIR) -> Tuple[Chroma, ParentDocumentRetriever]:
    """Build Chroma children index + parent retriever."""
    print("🔄  Rebuilding vector store …")

    # --- 2.1  parent docs = one Document per *article* ----------------- #
    md_text = Path(md_path).read_text(encoding="utf-8")

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADER_MAP,
        strip_headers=False,
    )
    parent_docs = header_splitter.split_text(md_text)

    # ─── Análise de cada artigo antes de indexar ──────────────────────────
    import re, tiktoken

    enc = tiktoken.encoding_for_model("gpt-4o-mini")  # usa mesmo encoder do modelo

    # parâmetros de corte
    _MIN_CHARS   = 80          # evita “CAPÍTULO I\nDA ZFM…”
    _MIN_TOKENS  = 40
    _DIGIT_RE    = re.compile(r"\d|§|%|\bart\.?\b", flags=re.I)

    def _flag_has_body(txt: str) -> bool:
        """True se o artigo contém corpo relevante (texto > cabeçalho)."""
        if len(txt.strip()) < _MIN_CHARS:
            return False
        if len(enc.encode(txt)) < _MIN_TOKENS:
            return False
        return bool(_DIGIT_RE.search(txt))   # precisa ter número, § ou %

    for d in parent_docs:
        txt = d.page_content

        d.metadata.update(
            {
                "lang": "pt-BR",
                "token_count": len(enc.encode(txt)),
                "has_body": _flag_has_body(txt),
            }
        )
        # cabeçalho-puro = quando NÃO tem corpo relevante
        d.metadata["header_only"] = not d.metadata["has_body"]

    # --- 2.2  child splitter ------------------------------------------- #
    child_splitter = RecursiveCharacterTextSplitter(**PARA_SPLITTER_KWARGS)

    # --- 2.3  vector store & doc-store --------------------------------- #
    shutil.rmtree(db_path, ignore_errors=True)          # clean slate
    embedding = OpenAIEmbeddings(model=EMBED_MODEL, dimensions=EMBED_DIM)

    vectordb = Chroma(
        collection_name="law214",
        embedding_function=embedding,
        persist_directory=db_path,
    )
    
    doc_store = InMemoryStore()           # parents live only for this run

    # --- 2.4  parent retriever builds children & indexes them ----------- #
    parent_retriever = ParentDocumentRetriever(
        vectorstore=vectordb,
        docstore=doc_store,
        child_splitter=child_splitter,
    )
    parent_retriever.add_documents(parent_docs)         # heavy lifting happens here


    print(f"✅  Indexed {vectordb._collection.count()} child chunks "
          f"mapping to {len(parent_docs)} parent articles")
    return vectordb



# --------------------------------------------------------------------------- #
# 3.  Retrieval helpers                                                       #
# --------------------------------------------------------------------------- #
def get_vectordb(db_path: str = CHROMA_DIR) -> Chroma:
    """Load an existing Chroma collection (error if missing)."""
    if not Path(db_path).exists():
        raise FileNotFoundError(
            "Vector store not found. Run with --reindex first."
        )
    embedding = OpenAIEmbeddings(model=EMBED_MODEL, dimensions=EMBED_DIM)
    return Chroma(
        collection_name="law214",
        embedding_function=embedding,
        persist_directory=db_path,
    )

# --------------------------------------------------------------------------- #
# 3.a  Load parent‑aware retriever                                            #
# --------------------------------------------------------------------------- #
def get_parent_retriever(db_path: str = CHROMA_DIR) -> ParentDocumentRetriever:
    """Return a ParentDocumentRetriever that yields whole articles."""
    embedding = OpenAIEmbeddings(model=EMBED_MODEL, dimensions=EMBED_DIM)
    vectordb = Chroma(
        collection_name="law214",
        embedding_function=embedding,
        persist_directory=db_path,
    )
    doc_store = InMemoryStore()           # parents live only for this run
    return ParentDocumentRetriever(
        vectorstore=vectordb,
        docstore=doc_store,
        child_splitter=RecursiveCharacterTextSplitter(**PARA_SPLITTER_KWARGS),
    )


# --------------------------------------------------------------------------- #
# 3.z  Low‑level helper: fetch & merge parents (shared by ask / QA)           #
# --------------------------------------------------------------------------- #
from langchain.docstore.document import Document

def retrieve_parents(
    query: str,
    vectordb: Chroma,
    *,
    k: int = DEFAULT_K_PARENTS,
    k_fetch: int = DEFAULT_K_FETCH,
    mmr: bool = False,
    lambda_mult: float = 0.8,
) -> List[Tuple[Document, float]]:
    """
    Core retrieval routine used by both `ask_question` (CLI print) and the
    custom QA retriever. Returns a list of (parent_doc, best_dist) pairs,
    truncated to *k* articles.  For MMR the distance is NaN.
    """
    # ---------- 1) child‑level fetch ----------------------------------- #
    if not mmr:
        # caminho antigo — distância fornecida pela API
        child_pairs = vectordb.similarity_search_with_score(
            query, k=k_fetch, filter={"has_body": True}
        )
    else:
        # ---------- (1)  MMR devolve só os documentos -----------------
        mmr_docs = vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k_fetch,
                "lambda_mult": lambda_mult,   # 1=similarity_search  0=pure diversity, ignoring the user query
                "filter": {"has_body": True},
            },
        ).invoke(query)

        # ---------- (2)  embed uma única vez o query ------------------
        embed = vectordb._embedding_function          # mesmo objeto usado pelo VS
        q_vec = np.asarray(embed.embed_query(query))

        # ---------- (3)  embed em lote os docs MMR -------------------
        d_texts = [d.page_content for d in mmr_docs]
        d_vecs  = embed.embed_documents(d_texts)      # lista[list[float]]

        # ---------- (4)  distância cosseno  →  mesmo “scale” do Chroma
        def cos_dist(a: np.ndarray, b: np.ndarray) -> float:
            return 1.0 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

        child_pairs = [
            (doc, cos_dist(q_vec, np.asarray(vec)))
            for doc, vec in zip(mmr_docs, d_vecs)
        ]

    # ---------- 2) merge children -> parents --------------------------- #
    from collections import defaultdict

    bucket: dict[str, list[Tuple[Document, float]]] = defaultdict(list)
    for doc, dist in child_pairs:
        key = doc.metadata.get("article") or f"HDR:{id(doc)}"
        bucket[key].append((doc, dist))

    parents: list[Tuple[Document, float]] = []
    for slices in bucket.values():
        slices.sort(key=lambda p: p[1])          # best slice first
        rep_doc, best_dist = slices[0]

        # de‑duplicate the overlap (use splitter config)
        overlap = PARA_SPLITTER_KWARGS.get("chunk_overlap", 0)
        parts   = [d.page_content for d, _ in slices]
        # remove exactly the overlap we introduced (and no more)
        text    = parts[0] + "".join(p[overlap:] for p in parts[1:])

        parent_doc = Document(page_content=text, metadata=rep_doc.metadata)
        parents.append((parent_doc, best_dist))

    # ---------- 3) rank & truncate ------------------------------------- #
    parents.sort(key=lambda p: p[1])             # NaNs last
    return parents[:k]


from langchain_core.retrievers import BaseRetriever

class ParentListRetriever(BaseRetriever):
    """Tiny wrapper so RetrievalQA can reuse the same parent selection."""
    def __init__(self, vectordb: Chroma, *, mmr: bool, lambda_mult: float = 0.8):
        super().__init__()
        self._db = vectordb
        self._mmr = mmr
        self._lambda_mult = lambda_mult

    def _get_relevant_documents(self, query: str, *, run_manager=None, **kwargs):
        parents = retrieve_parents(
            query, self._db, mmr=self._mmr, lambda_mult=self._lambda_mult
        )
        return [d for d, _ in parents]


def ask_question(
    query: str,
    vectordb: Chroma,
    *,
    k: int = DEFAULT_K_PARENTS,
    k_fetch: int = DEFAULT_K_FETCH,
    mmr: bool = False,
    lambda_mult: float = 0.8,
) -> List[Tuple[float, str, str]]:
    """
    Fetch *k_fetch* child chunks, merge them into parent articles,
    then keep the top *k* parents.

    Pass both knobs explicitly via the function signature.

    • similarity path -> keeps real Euclidean distance (best child slice)
    • mmr path        -> distance is NaN (MMR API has no score)
    """

    parents = retrieve_parents(
        query,
        vectordb,
        k=k,
        k_fetch=k_fetch,
        mmr=mmr,
        lambda_mult=lambda_mult,
    )

    def path(meta: dict) -> str:
        keys = ("book", "title", "chapter", "section", "article")
        return " > ".join(meta[k] for k in keys if k in meta)

    results = []
    for doc, dist in parents:
        results.append((dist, path(doc.metadata), doc.page_content))
    return results



# --------------------------------------------------------------------------- #
# >>  optional: build a RetrievalQA chain on demand                           #
# --------------------------------------------------------------------------- #
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

# ----------------------------------- PROMPT ------------------------------------------------ #
PROMPT_TXT = """
Você é tributarista especializado na Lei Complementar 214/2025
(IBS, CBS e IS).  
Responda **apenas** com base nos trechos abaixo; use conhecimento externo apenas sobre temas diretamente tratados na lei complementar 214/2025.  
Se algum dado (percentual, artigo, inciso, parágrafo) não aparecer
explicitamente, diga **“Não localizado”**.  
Nunca invente números ou fundamentos jurídicos.
Sempre que possível, deixar claro quando a tributação for vincula às aquisições/entradas ou vendas/saídas.
Formato da resposta  
• Português formal  
• máx. 250 palavras  
• Itens numerados:  
  1. <descrição detalhada> — <percentual ou termo-chave>, citação (Art. ###, § #, I, …)  


{context}

Pergunta: {question}
"""
# ------------------------------------------------------------------------------------------- #

def get_qa_chain(*, mmr: bool = False, lambda_mult: float = 0.8) -> RetrievalQA:
    """Return a RetrievalQA chain that shares retrieval logic with CLI."""
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    vectordb = get_vectordb()                     # load existing children index
    qa_ret   = ParentListRetriever(
        vectordb, mmr=mmr, lambda_mult=lambda_mult
    )

    # cria o template UMA ÚNICA VEZ
    PROMPT = PromptTemplate.from_template(PROMPT_TXT)

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=qa_ret,
        chain_type_kwargs={"prompt": PROMPT},
    )




# --------------------------------------------------------------------------- #
# 4.  CLI                                                                     #
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG helper for Lei Complementar 214/2025 (IBS/CBS/IS)."
    )
    parser.add_argument("--reindex", action="store_true",
                        help="(re)build the vector store from markdown.")
    parser.add_argument("--ask", metavar="QUERY",
                        help="Run a similarity/MRR search.")
    parser.add_argument("--mmr", action="store_true",
                        help="Use MMR hybrid search instead of pure similarity.")
    parser.add_argument("--qa", action="store_true",
                        help="Use RetrievalQA chain to ask and get a synthesized answer.")

    args = parser.parse_args()

    if args.reindex:
        vectordb = build_vectorstore()
    else:
        vectordb = get_vectordb()

    parent_retriever = get_parent_retriever()

    # ------------------------------------------------------------------ #
    #  Handle interactive query                                          #
    # ------------------------------------------------------------------ #
    if args.ask and args.qa:
        # Option 1: synthesized answer via RetrievalQA
        qa_chain = get_qa_chain(mmr=args.mmr)
        response = qa_chain.invoke({"query": args.ask})
        print("\n=== Resposta Sintetizada ===\n")
        print(response["result"])

    elif args.ask:
        # Default: show raw passages
        for dist, fullpath, snippet in ask_question(
            args.ask,
            parent_retriever.vectorstore,
            mmr=args.mmr,
        ):
            # Handle NaN distances returned by MMR path
            if math.isnan(dist):
                sim_text  = "—"
                dist_text = "—"
            else:
                sim_text  = f"{1 / (1 + dist):0.3f}"
                dist_text = f"{dist:0.3f}"

            print(f"\n{sim_text}  (d={dist_text})  {fullpath}\n{snippet}\n")


if __name__ == "__main__":
    main()