"""Config-driven document ingestion pipeline.

This refactors the legacy ingest script to accept an agent configuration
that defines chunking, embedding, and model settings.
"""

import argparse
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI
from supabase import Client, create_client
from tqdm import tqdm

from data.document import DocumentPage, PDFDocument, create_document
from ops.agent_utils import build_chunker, load_agent_config, resolve_model_section
from ops.chunking.chunker import Chunker


# Ensure environment variables (Supabase/OpenAI keys) are loaded.
load_dotenv(find_dotenv(usecwd=True))


def pdf_pages(path: str) -> Iterable[DocumentPage]:
    """Yield cleaned PDF pages using the document abstraction."""

    document = create_document(path)
    if not isinstance(document, PDFDocument):
        raise TypeError(f"Expected a PDFDocument for path '{path}', got {type(document)!r}")

    for page in document.iter_pages():
        yield page
def chunk_document(chunker: Chunker, pages: Iterable[Any]) -> List[Dict[str, Any]]:
    page_payload = []
    for page in pages:
        if isinstance(page, DocumentPage):
            page_data = page.to_dict()
            if "page_number" in page_data:
                page_data["page_number"] = page.page_number + 1
            page_payload.append(page_data)
        else:
            page_payload.append(page)
    return chunker.chunk_pages(page_payload)


def create_supabase_client() -> Client:
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    return create_client(url, key)


def create_openai_client() -> OpenAI:
    api_key = os.environ["OPENAI_API_KEY"]
    return OpenAI(api_key=api_key)


def resolve_input_paths(path_or_dir: str) -> List[Path]:
    """Return a sorted list of PDF file paths from a single path or directory."""

    candidate = Path(path_or_dir)
    if not candidate.exists():
        raise FileNotFoundError(f"Input path does not exist: {candidate}")

    if candidate.is_file():
        if candidate.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file, got: {candidate}")
        return [candidate]

    if candidate.is_dir():
        pdfs = sorted(
            p for p in candidate.rglob("*.pdf") if p.is_file()
        )
        if not pdfs:
            raise FileNotFoundError(f"No PDF files found under directory: {candidate}")
        return pdfs

    raise ValueError(f"Unsupported input path: {candidate}")


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-")
    return slug.lower() or "document"


def ingest_single_document(
    pdf_path: Path,
    chunker: Chunker,
    embed_model: str,
    embed_batch: int,
    supabase_client: Client,
    openai_client: OpenAI,
    table_name: str,
    batch_insert: int,
    clear_existing: bool,
    base_doc_id: str,
) -> Tuple[str, int]:
    """Ingest a single PDF file and return (doc_id, rows_inserted)."""

    doc_id = base_doc_id
    if "{name}" in base_doc_id:
        doc_id = base_doc_id.format(name=pdf_path.stem)

    if clear_existing:
        supabase_client.table(table_name).delete().eq("doc_id", doc_id).execute()

    print(f"Loading PDF from {pdf_path}...")
    pages = list(pdf_pages(str(pdf_path)))

    chunker_cfg = chunker.config if hasattr(chunker, "config") else {}
    strategy_name = getattr(chunker, "__class__", type(chunker)).__name__
    chunk_mode = chunker_cfg.get("chunk_mode") if isinstance(chunker_cfg, dict) else None
    if chunk_mode:
        print(f"Chunking with strategy '{strategy_name}' ({chunk_mode} mode)...")
    else:
        print(f"Chunking with strategy '{strategy_name}'...")

    chunk_records = chunk_document(chunker, pages)
    inputs = [record["chunk_text"] for record in chunk_records]
    metas = [
        {
            "page": record.get("page_number"),
            "source": str(pdf_path),
        }
        for record in chunk_records
    ]

    print(f"✅ Built {len(inputs)} chunks from {pdf_path}")

    vectors: List[List[float]] = []
    for idx in tqdm(range(0, len(inputs), embed_batch), desc=f"Embedding {pdf_path.name}"):
        batch_inputs = inputs[idx : idx + embed_batch]
        if not batch_inputs:
            continue
        response = openai_client.embeddings.create(model=embed_model, input=batch_inputs)
        vectors.extend([item.embedding for item in response.data])

    rows = []
    for idx, (content, embedding, meta) in enumerate(zip(inputs, vectors, metas)):
        rows.append(
            {
                "doc_id": doc_id,
                "chunk_index": idx,
                "content": content,
                "metadata": meta,
                "embedding": embedding,
            }
        )

    for start in tqdm(range(0, len(rows), batch_insert), desc=f"Uploading {pdf_path.name}"):
        supabase_client.table(table_name).insert(rows[start : start + batch_insert]).execute()

    print(f"🎉 Done! Inserted {len(rows)} chunks for doc_id={doc_id}")
    return doc_id, len(rows)


def ingest(config_path: str) -> None:
    agent_cfg = load_agent_config(config_path)

    data_cfg = agent_cfg.get("data_source", {})
    path_or_dir = data_cfg.get("path")
    if not path_or_dir:
        raise ValueError("data_source.path must be provided in the agent config")

    input_paths = resolve_input_paths(path_or_dir)

    base_doc_id = agent_cfg.get("doc_id")
    if not base_doc_id:
        config_slug = _slugify(Path(config_path).stem)
        if Path(path_or_dir).is_dir() or len(input_paths) > 1:
            base_doc_id = f"{config_slug}-{{name}}"
        else:
            base_doc_id = f"{config_slug}-docs"
        print(f"Using inferred doc_id template '{base_doc_id}'")

    supabase_cfg = agent_cfg.get("supabase", {})
    table_name = supabase_cfg.get("table", "chunks")
    batch_insert = supabase_cfg.get("batch_insert", 200)
    clear_existing = supabase_cfg.get("clear_existing", True)

    chunker = build_chunker(agent_cfg.get("chunker"))
    embed_cfg = resolve_model_section(agent_cfg.get("embed_model"))
    embed_model = embed_cfg.get("name")
    if not embed_model:
        raise ValueError("embed_model.name (or model_zoo_id) must be provided")
    embed_batch = embed_cfg.get("batch_size", 100)

    model_cfg = resolve_model_section(agent_cfg.get("model", {}))
    if model_cfg:
        print(
            f"Resolved model '{model_cfg.get('name')}' from model zoo (not used during ingest)."
        )

    supabase_client = create_supabase_client()
    openai_client = create_openai_client()

    total_rows = 0
    for pdf_path in input_paths:
        doc_id, rows = ingest_single_document(
            pdf_path=pdf_path,
            chunker=chunker,
            embed_model=embed_model,
            embed_batch=embed_batch,
            supabase_client=supabase_client,
            openai_client=openai_client,
            table_name=table_name,
            batch_insert=batch_insert,
            clear_existing=clear_existing,
            base_doc_id=base_doc_id,
        )
        total_rows += rows

    print(f"🎉 Completed ingest of {len(input_paths)} documents. Total rows inserted: {total_rows}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest documents into Supabase using an agent config")
    parser.add_argument(
        "--config",
        default="configs/agents/nutrition_ingest.yaml",
        help="Path to the agent YAML configuration file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ingest(args.config)
