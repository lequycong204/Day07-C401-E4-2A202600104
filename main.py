from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from src.chunking import RecursiveChunker
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from src.models import Document
from src.store import EmbeddingStore

SAMPLE_FILES = [
    "data/ai_engineer.md",
]

CHROMA_PERSIST_DIR = Path("data/chroma_ai_engineer_db")
CHROMA_COLLECTION_NAME = "ai_engineer_chunks"
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_TOP_K = 5


def _split_markdown_pages(content: str) -> list[tuple[int | None, str]]:
    pattern = re.compile(r"<!--\s*page:\s*(\d+)\s*-->")
    matches = list(pattern.finditer(content))
    if not matches:
        stripped = content.strip()
        return [(None, stripped)] if stripped else []

    pages: list[tuple[int | None, str]] = []
    for index, match in enumerate(matches):
        page_id = int(match.group(1))
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
        page_content = content[start:end].strip()
        if page_content:
            pages.append((page_id, page_content))
    return pages


def _chunk_file_content(path: Path, content: str, chunker: RecursiveChunker) -> list[Document]:
    documents: list[Document] = []
    page_blocks = _split_markdown_pages(content) if path.suffix.lower() == ".md" else [(None, content.strip())]

    for page_id, page_content in page_blocks:
        if not page_content:
            continue

        chunks = chunker.chunk(page_content)
        for chunk_index, chunk_text in enumerate(chunks, start=1):
            doc_suffix = page_id if page_id is not None else "na"
            documents.append(
                Document(
                    id=f"{path.stem}_p{doc_suffix}_c{chunk_index}",
                    content=chunk_text,
                    metadata={
                        "source": str(path),
                        "extension": path.suffix.lower(),
                        "page_id": page_id,
                        "chunk_id": chunk_index,
                    },
                )
            )
    return documents


def load_documents_from_files(file_paths: list[str], chunk_size: int = DEFAULT_CHUNK_SIZE) -> list[Document]:
    allowed_extensions = {".md", ".txt"}
    documents: list[Document] = []
    chunker = RecursiveChunker(chunk_size=chunk_size)

    for raw_path in file_paths:
        path = Path(raw_path)
        if path.suffix.lower() not in allowed_extensions:
            continue
        if not path.exists() or not path.is_file():
            continue

        content = path.read_text(encoding="utf-8")
        documents.extend(_chunk_file_content(path, content, chunker))

    return documents


def build_embedder() -> object:
    load_dotenv(override=False)
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()

    if provider == "local":
        try:
            return LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception:
            return _mock_embed

    if provider == "openai":
        try:
            return OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
        except Exception:
            return _mock_embed

    return _mock_embed


def build_store(embedder: object) -> EmbeddingStore:
    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    return EmbeddingStore(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_fn=embedder,  # type: ignore[arg-type]
        persist_directory=str(CHROMA_PERSIST_DIR),
    )


def ingest_documents_if_needed(store: EmbeddingStore, docs: list[Document]) -> tuple[bool, int]:
    current_size = store.get_collection_size()
    if current_size > 0:
        return False, current_size

    store.add_documents(docs)
    return True, store.get_collection_size()


def format_context(chunks: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata", {})
        blocks.append(
            "\n".join(
                [
                    f"[{index}] source={metadata.get('source')}",
                    f"page_id={metadata.get('page_id')} chunk_id={metadata.get('chunk_id')} score={chunk.get('score', 0):.3f}",
                    chunk.get("content", ""),
                ]
            )
        )
    return "\n\n".join(blocks)


def call_gpt_5_nano(prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "[ERROR] OPENAI_API_KEY is not set."

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model="gpt-5-nano",
            instructions=(
                "You are a concise retrieval-augmented assistant. "
                "Answer only from the provided context. "
                "If the context is insufficient, say so clearly. "
                "When useful, mention page_id and chunk_id."
            ),
            input=prompt,
            max_output_tokens=1200,
        )
        return response.output_text or ""
    except Exception as exc:
        return f"[ERROR calling gpt-5-nano] {exc}"


def search_chunks(
    store: EmbeddingStore,
    query: str,
    top_k: int,
    metadata_filter: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if metadata_filter:
        return store.search_with_filter(query, top_k=top_k, metadata_filter=metadata_filter)
    return store.search(query, top_k=top_k)


def answer_query(
    store: EmbeddingStore,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    metadata_filter: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    chunks = search_chunks(store, query, top_k=top_k, metadata_filter=metadata_filter)
    if not chunks:
        return [], "No relevant chunks were found."

    context = format_context(chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    return chunks, call_gpt_5_nano(prompt)


def parse_metadata_filter(filter_key: str, filter_value: str) -> dict[str, Any] | None:
    if not filter_key or not filter_value:
        return None
    normalized_value: Any = filter_value
    if filter_value.isdigit():
        normalized_value = int(filter_value)
    return {filter_key: normalized_value}


@st.cache_resource(show_spinner=False)
def get_embedder() -> object:
    return build_embedder()


@st.cache_resource(show_spinner=False)
def get_store() -> EmbeddingStore:
    return build_store(get_embedder())


@st.cache_data(show_spinner=False)
def get_chunked_documents(file_paths: tuple[str, ...], chunk_size: int) -> list[Document]:
    return load_documents_from_files(list(file_paths), chunk_size=chunk_size)


def render_sidebar() -> tuple[list[str], int, int, dict[str, Any] | None]:
    st.sidebar.header("Cau hinh")
    files_raw = st.sidebar.text_area(
        "Danh sach file",
        value="\n".join(SAMPLE_FILES),
        help="Moi dong la mot file .md hoac .txt",
    )
    chunk_size = st.sidebar.number_input("chunk_size", min_value=200, max_value=4000, value=DEFAULT_CHUNK_SIZE, step=100)
    top_k = st.sidebar.number_input("top_k", min_value=1, max_value=20, value=DEFAULT_TOP_K, step=1)

    st.sidebar.subheader("Metadata filter")
    filter_key = st.sidebar.text_input("Filter key", value="")
    filter_value = st.sidebar.text_input("Filter value", value="")
    metadata_filter = parse_metadata_filter(filter_key.strip(), filter_value.strip())

    files = [line.strip() for line in files_raw.splitlines() if line.strip()]
    return files, int(chunk_size), int(top_k), metadata_filter


def render_retrieved_chunks(chunks: list[dict[str, Any]]) -> None:
    st.subheader("Retrieved chunks")
    if not chunks:
        st.info("Khong tim thay chunk phu hop.")
        return

    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata", {})
        with st.expander(
            f"{index}. score={chunk.get('score', 0):.3f} | page_id={metadata.get('page_id')} | chunk_id={metadata.get('chunk_id')}",
            expanded=index == 1,
        ):
            st.caption(f"source: {metadata.get('source')}")
            st.write(chunk.get("content", ""))


def main() -> None:
    st.set_page_config(page_title="AI Engineer RAG Demo", layout="wide")
    st.title("AI Engineer RAG Demo")
    st.write("Doc markdown, chunk theo page marker, luu vao ChromaDB va truy van bang GPT-5-nano.")

    files, chunk_size, top_k, metadata_filter = render_sidebar()
    docs = get_chunked_documents(tuple(files), chunk_size)

    col1, col2, col3 = st.columns(3)
    col1.metric("So chunk da tao", len(docs))
    col2.metric("chunk_size", chunk_size)
    col3.metric("top_k", top_k)

    embedder = get_embedder()
    st.caption(f"Embedding backend: {getattr(embedder, '_backend_name', embedder.__class__.__name__)}")

    store = get_store()

    if st.button("Nap vao / Load ChromaDB", type="primary"):
        with st.spinner("Dang khoi tao collection..."):
            indexed, count = ingest_documents_if_needed(store, docs)
        if indexed:
            st.success(f"Da index {count} chunks vao ChromaDB.")
        else:
            st.info(f"Da load collection ton tai voi {count} chunks.")

    st.write(f"ChromaDB hien co: `{store.get_collection_size()}` chunks")

    query = st.text_area(
        "Nhap query",
        value="Tom tat cac y chinh cua tai lieu nay.",
        height=120,
    )

    if metadata_filter:
        st.caption(f"Dang dung metadata_filter = {metadata_filter}")

    if st.button("Truy van", use_container_width=True):
        if store.get_collection_size() == 0:
            st.warning("Collection dang rong. Hay bam 'Nap vao / Load ChromaDB' truoc.")
            return
        if not query.strip():
            st.warning("Hay nhap query.")
            return

        with st.spinner("Dang truy van va goi LLM..."):
            chunks, answer = answer_query(
                store,
                query.strip(),
                top_k=top_k,
                metadata_filter=metadata_filter,
            )
        render_retrieved_chunks(chunks)
        st.subheader("LLM answer")
        st.write(answer)

    st.divider()
    st.subheader("Goi y cho Query 2")
    st.code('metadata_filter = {"page_id": 1}', language="python")
    st.caption("Schema hien tai co `page_id`, `chunk_id`, `source`, `extension`, `doc_id`. Neu muon filter theo Chapter 1, can map chapter -> page range hoac them field `part/chapter` luc index.")


if __name__ == "__main__":
    main()

