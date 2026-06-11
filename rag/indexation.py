#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

"""
build_faiss_rag_db.py

Crée une base vectorielle FAISS à partir des fichiers chunks JSONL générés par :
- prepare_corpus.py
- translate_corpus.py

Entrées possibles :
- corpus/chunks.jsonl
- corpus_translated/chunks_en.jsonl
- corpus_translated/chunks_gcf.jsonl

Sortie :
- un dossier FAISS local, par exemple vectorstore_mici/

Installation :
pip install -U langchain langchain-community langchain-huggingface sentence-transformers faiss-cpu tqdm

Exemple :
python rag/indexation.py \
  --chunks-files ./corpus/chunks.jsonl ./corpus_translated/chunks_en.jsonl ./corpus_translated/chunks_gcf.jsonl \
  --output-dir ./vectorstore_mici \
  --embedding-model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
  --overwrite \
  --test-query "Que faire en cas de fatigue avec une MICI ?"
"""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from tqdm import tqdm


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"JSON invalide dans {path}, ligne {line_number}: {e}"
                ) from e

    return rows


def infer_language_from_file(path: Path) -> str | None:
    name = path.name.lower()

    if name.endswith("_en.jsonl"):
        return "en"

    if name.endswith("_gcf.jsonl"):
        return "gcf"

    if name == "chunks.jsonl":
        return "fr"

    return None


def row_to_document(row: dict, fallback_language: str | None = None):
    """
    Convertit une ligne JSONL en Document LangChain.
    """
    from langchain_core.documents import Document

    text = str(row.get("text", "")).strip()

    if not text:
        return None

    language = row.get("language") or row.get("target_language") or fallback_language

    metadata = {
        "id": row.get("id"),
        "source_id": row.get("source_id"),
        "document_id": row.get("document_id"),
        "chunk_index": row.get("chunk_index"),
        "source_file": row.get("source_file"),
        "source_path": row.get("source_path"),
        "text_path": row.get("text_path"),
        "modality": row.get("modality"),
        "language": language,
        "target_language": row.get("target_language"),
        "source_language": row.get("source_language"),
        "translation_model": row.get("translation_model"),
    }

    metadata = {
        key: value
        for key, value in metadata.items()
        if value is not None and isinstance(value, (str, int, float, bool))
    }

    return Document(
        page_content=text,
        metadata=metadata,
    )


def load_documents_from_chunks_files(chunks_files: list[Path]) -> list:
    documents = []
    seen_ids = set()

    for chunks_file in chunks_files:
        if not chunks_file.exists():
            raise FileNotFoundError(f"Fichier introuvable : {chunks_file}")

        fallback_language = infer_language_from_file(chunks_file)
        rows = load_jsonl(chunks_file)

        print(f"Chargé : {chunks_file} — {len(rows)} chunks")

        for row in tqdm(rows, desc=f"Conversion {chunks_file.name}"):
            doc = row_to_document(row, fallback_language=fallback_language)

            if doc is None:
                continue

            doc_id = doc.metadata.get("id")

            if doc_id and doc_id in seen_ids:
                continue

            if doc_id:
                seen_ids.add(doc_id)

            documents.append(doc)

    return documents


def atomic_replace_dir(src_dir: Path, dst_dir: Path) -> None:
    backup_dir = None

    if dst_dir.exists():
        backup_dir = dst_dir.with_name(dst_dir.name + ".bak")

        if backup_dir.exists():
            shutil.rmtree(backup_dir)

        os.replace(dst_dir, backup_dir)

    try:
        os.replace(src_dir, dst_dir)

        if backup_dir and backup_dir.exists():
            shutil.rmtree(backup_dir)

    except Exception:
        if dst_dir.exists():
            shutil.rmtree(dst_dir)

        if backup_dir and backup_dir.exists():
            os.replace(backup_dir, dst_dir)

        raise


def build_faiss_index(
    documents: list,
    output_dir: Path,
    embedding_model: str,
    batch_size: int,
    device: str,
    normalize_embeddings: bool,
    overwrite: bool,
    source_files: list[Path] | None = None,
) -> None:
    if not documents:
        raise ValueError("Aucun document à indexer.")
    if batch_size < 1:
        raise ValueError("batch_size doit être supérieur ou égal à 1.")

    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Le dossier existe déjà : {output_dir}. "
            "Utilise --overwrite pour le remplacer."
        )

    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    print("\nChargement du modèle d'embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={
            "device": device,
        },
        encode_kwargs={
            "batch_size": batch_size,
            "normalize_embeddings": normalize_embeddings,
        },
    )

    print("\nCréation de la base vectorielle FAISS...")
    print(f"Documents à indexer : {len(documents)}")
    print(f"Modèle embeddings : {embedding_model}")
    print(f"Device : {device}")
    print(f"Batch size : {batch_size}")

    db = FAISS.from_documents(documents, embeddings)

    tmp_parent = output_dir.parent
    tmp_parent.mkdir(parents=True, exist_ok=True)

    tmp_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.",
            suffix=".tmp",
            dir=str(tmp_parent),
        )
    )

    try:
        db.save_local(str(tmp_dir))

        manifest = {
            "embedding_model": embedding_model,
            "documents_count": len(documents),
            "device": device,
            "batch_size": batch_size,
            "normalize_embeddings": normalize_embeddings,
            "source_files": [
                {
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                }
                for path in (source_files or [])
            ],
        }

        (tmp_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        atomic_replace_dir(tmp_dir, output_dir)

    except Exception:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        raise

    print(f"\nBase FAISS créée : {output_dir}")
    print(f"{len(documents)} chunks indexés.")


def test_search(
    output_dir: Path,
    embedding_model: str,
    query: str,
    k: int,
    device: str,
    normalize_embeddings: bool,
) -> None:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={
            "device": device,
        },
        encode_kwargs={
            "normalize_embeddings": normalize_embeddings,
        },
    )

    db = FAISS.load_local(
        str(output_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    results = db.similarity_search_with_score(query, k=k)

    print("\nTest de recherche")
    print(f"Question : {query}")

    for i, (doc, score) in enumerate(results, start=1):
        metadata = doc.metadata
        preview = doc.page_content.replace("\n", " ")[:300]

        print(f"\n[{i}] score={score}")
        print(f"source_file={metadata.get('source_file')}")
        print(f"language={metadata.get('language')}")
        print(f"modality={metadata.get('modality')}")
        print(f"chunk_index={metadata.get('chunk_index')}")
        print(preview)


def main():
    parser = argparse.ArgumentParser(
        description="Construit une base RAG FAISS à partir de fichiers chunks JSONL."
    )

    parser.add_argument(
        "--chunks-files",
        type=Path,
        nargs="+",
        required=True,
        help="Fichiers chunks JSONL à indexer."
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Dossier de sortie FAISS."
    )

    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Modèle Hugging Face pour les embeddings."
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size pour le calcul des embeddings."
    )

    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device pour les embeddings."
    )

    parser.add_argument(
        "--no-normalize-embeddings",
        action="store_true",
        help="Désactive la normalisation des embeddings."
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remplace le vectorstore s'il existe déjà."
    )

    parser.add_argument(
        "--test-query",
        default=None,
        help="Question de test à exécuter après création de la base."
    )

    parser.add_argument(
        "--test-k",
        type=int,
        default=5,
        help="Nombre de résultats pour le test de recherche."
    )

    args = parser.parse_args()

    documents = load_documents_from_chunks_files(args.chunks_files)

    build_faiss_index(
        documents=documents,
        output_dir=args.output_dir,
        embedding_model=args.embedding_model,
        batch_size=args.batch_size,
        device=args.device,
        normalize_embeddings=not args.no_normalize_embeddings,
        overwrite=args.overwrite,
        source_files=args.chunks_files,
    )

    if args.test_query:
        test_search(
            output_dir=args.output_dir,
            embedding_model=args.embedding_model,
            query=args.test_query,
            k=args.test_k,
            device=args.device,
            normalize_embeddings=not args.no_normalize_embeddings,
        )


if __name__ == "__main__":
    main()
