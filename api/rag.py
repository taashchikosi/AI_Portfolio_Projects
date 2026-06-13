"""Shared RAG infrastructure for all four agents.

- Chat LLM: DeepSeek (OpenAI-compatible API).
- Embeddings: OpenAI (MANDATORY) -- the Chroma stores were built with
  OpenAIEmbeddings and must be queried in the same embedding space.
"""
import os
from functools import lru_cache

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


@lru_cache(maxsize=1)
def load_llm():
    # DeepSeek is OpenAI-compatible: same client, different base_url + key.
    return ChatOpenAI(
        model="deepseek-chat",
        temperature=0,
        base_url="https://api.deepseek.com",
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
    )


@lru_cache(maxsize=1)
def load_embeddings():
    # Must stay OpenAI -- vector stores live in OpenAI's embedding space.
    return OpenAIEmbeddings()


@lru_cache(maxsize=None)
def load_vectordb(persist_dir: str):
    return Chroma(persist_directory=persist_dir, embedding_function=load_embeddings())


def keys_ready() -> bool:
    """RAG needs DeepSeek (chat) AND OpenAI (embeddings)."""
    return bool(os.environ.get("DEEPSEEK_API_KEY")) and bool(os.environ.get("OPENAI_API_KEY"))


def rag_answer(persist_dir, system_prompt: str, question: str, k: int = 5) -> str:
    vectordb = load_vectordb(str(persist_dir))
    hits = vectordb.similarity_search(question, k=k)

    context_parts = []
    for i, h in enumerate(hits, start=1):
        src = h.metadata.get("source", "kb_docs")
        chunk = h.metadata.get("chunk", i)
        context_parts.append(f"[SOURCE: {src} | CHUNK: {chunk}]\n{h.page_content}")
    context = "\n\n".join(context_parts)

    prompt = f"QUESTION:\n{question}\n\nCONTEXT:\n{context}"
    llm = load_llm()
    return llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
    ).content
