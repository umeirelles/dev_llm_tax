#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit UI – Lei Complementar 214/2025 RAG demo
Compatível com llm_langchain_test.py (mai 2025).

Requisitos:
  pip install streamlit pandas
  # + dependências que você já tem no projeto
"""

from pathlib import Path
import math
import os

import streamlit as st
import pandas as pd

# ────────────────────────────────────────────────────────────────────────────
# Backend helpers (importados do seu módulo)
# ────────────────────────────────────────────────────────────────────────────
from llm_langchain_test import (
    get_vectordb,
    ask_question,
    get_qa_chain,
    DEFAULT_K_PARENTS,
)

# ────────────────────────────────────────────────────────────────────────────
# 1. Configuração da página
# ────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Lei 214 • RAG demo",
    page_icon="⚖️",
    layout="centered",
)

st.title("Pergunte sobre a Lei Complementar 214/2025 (IBS + CBS + IS)")

# ────────────────────────────────────────────────────────────────────────────
# 2. Carrega (ou avisa) o vetor-store
# ────────────────────────────────────────────────────────────────────────────
try:
    vectordb = get_vectordb()
except FileNotFoundError:
    vectordb = None
    st.error(
        "❌  Vetor-store não encontrado.\n\n"
        "Abra um terminal no diretório do projeto e execute:\n\n"
        "```bash\npython llm_langchain_test.py --reindex\n```"
    )
    st.stop()

# ────────────────────────────────────────────────────────────────────────────
# 3. Sidebar – parâmetros
# ────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️  Knobs - Recall LLM")

    k_hits = st.slider(
        "Artigos retornados (k)",
        min_value=1,
        max_value=20,
        value=DEFAULT_K_PARENTS,
    )
    use_mmr = st.checkbox("Buscar via MMR (híbrido denso + lexical)", value=False)

    lambda_mult = st.slider(
        "λ  (lambda_mult) — equilíbrio sim/diversidade",
        min_value=0.0,
        max_value=1.0,
        value=0.8,
        step=0.05,
    )

    want_answer = st.checkbox("Gerar resposta sintetizada (LLM)", value=True)

    st.divider()
    st.markdown(
        """
        *Menor distância → mais próximo*  
        Similaridade = `1 / (1 + dist)`
        """
    )

# ────────────────────────────────────────────────────────────────────────────
# 4. Caixa de pergunta
# ────────────────────────────────────────────────────────────────────────────
query = st.text_input(
    "Sua pergunta:",
    placeholder="Ex.: Quais alíquotas IBS incidem sobre exportações?",
)

if not query:
    st.stop()

# ────────────────────────────────────────────────────────────────────────────
# 5. Busca de passagens
# ────────────────────────────────────────────────────────────────────────────
hits = ask_question(
    query=query,
    vectordb=vectordb,
    k=k_hits,
    mmr=use_mmr,
    lambda_mult=lambda_mult,
)

if not hits:
    st.warning("Nenhuma passagem encontrada.")
    st.stop()

# DataFrame para exibir/baixar
df = pd.DataFrame(
    [
        {
            "distância": f"{d:.3f}" if not math.isnan(d) else "—",
            "similaridade": "—" if math.isnan(d) else f"{1 / (1 + d):.3f}",
            "local": path,
            "trecho": snippet,
        }
        for d, path, snippet in hits
    ]
)

st.markdown("### 🔎 Passagens selecionadas")

for dist, path, snippet in hits:
    sim = "—" if math.isnan(dist) else f"{1/(1+dist):.3f}"
    dist_txt = "—" if math.isnan(dist) else f"{dist:.3f}"
    st.markdown(
        f"""
        **Distância:** {dist_txt}  **Similaridade:** {sim}  
        **Local:** `{path}`  
        """,
        unsafe_allow_html=True,
    )
    # mostra o trecho sem blockquote ↓
    st.markdown(snippet)
    st.divider()  

# ────────────────────────────────────────────────────────────────────────────
# 6. Resposta sintetizada opcional
# ────────────────────────────────────────────────────────────────────────────
if want_answer:
    with st.spinner("Gerando resposta…"):
        qa_chain = get_qa_chain(mmr=use_mmr, lambda_mult=lambda_mult)
        resp = qa_chain.invoke({"query": query})
        answer = resp["result"]

    st.subheader("🧠 Resposta sintetizada")
    st.write(answer)

# ────────────────────────────────────────────────────────────────────────────
# 7. Rodapé
# ────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Demo RAG • LangChain • "
    "Desenvolvido por Ubaldino Meirelles"
)