#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
prepare_corpus.py

Prépare un corpus RAG à partir de vidéos, images et PDFs.

Fonctionnalités :
- transcription des vidéos avec Whisper ;
- OCR des images avec Tesseract ;
- extraction texte des PDFs avec fallback OCR page par page ;
- parallélisation images / PDFs / vidéos ;
- reprise automatique après interruption ;
- écriture atomique des .txt et .meta.json ;
- génération de documents.jsonl et chunks.jsonl ;
- suivi du progrès avec tqdm ;
- log d'exécution dans processing_log.jsonl.

Installation Colab recommandée :
!apt-get update -qq
!apt-get install -y ffmpeg tesseract-ocr tesseract-ocr-fra
!pip install -U openai-whisper pytesseract pillow tqdm pymupdf
"""

import argparse
import hashlib
import json
import os
import re
import tempfile
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from multiprocessing import cpu_count
from pathlib import Path

from PIL import Image, ImageOps, ImageFilter, ImageSequence
import pytesseract
from tqdm import tqdm


VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
PDF_EXTENSIONS = {".pdf"}

_WHISPER_MODEL = None


# ---------------------------------------------------------------------
# Utilitaires généraux
# ---------------------------------------------------------------------

def safe_filename_stem(path: Path) -> str:
    """
    Transforme un nom de fichier en nom stable.
    Exemple : 'Absences pour soins médicaux.png' -> 'Absences_pour_soins_medicaux'
    """
    name = unicodedata.normalize("NFKD", path.stem)
    name = name.encode("ascii", "ignore").decode("ascii")
    name = re.sub(r"[^a-zA-Z0-9_-]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "document"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)

    return h.hexdigest()


def iter_files(directory: Path, extensions: set[str]) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Le dossier n'existe pas : {directory}")

    return sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    )


def atomic_write_text(path: Path, text: str) -> None:
    """
    Écriture atomique :
    - écrit dans un fichier temporaire ;
    - fsync ;
    - remplace le fichier final seulement quand l'écriture est complète.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent)
    )

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text.strip() + "\n")
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_name, path)

    except Exception:
        try:
            os.remove(tmp_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent)
    )

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_name, path)

    except Exception:
        try:
            os.remove(tmp_name)
        except FileNotFoundError:
            pass
        raise


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent)
    )

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_name, path)

    except Exception:
        try:
            os.remove(tmp_name)
        except FileNotFoundError:
            pass
        raise


def write_event(log_path: Path, event_type: str, payload: dict) -> None:
    event = {
        "event": event_type,
        "time": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    append_jsonl(log_path, event)


# ---------------------------------------------------------------------
# Reprise après interruption
# ---------------------------------------------------------------------

def is_already_processed(source_path: Path, text_path: Path, meta_path: Path) -> bool:
    """
    Un fichier est considéré comme traité uniquement si :
    - le .txt existe ;
    - le .meta.json existe ;
    - le hash du fichier source correspond ;
    - le texte n'est pas vide.
    """
    if not text_path.exists() or not meta_path.exists():
        return False

    if text_path.stat().st_size == 0:
        return False

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    current_hash = sha256_file(source_path)

    return (
        meta.get("status") == "done"
        and meta.get("source_sha256") == current_hash
        and meta.get("text_path") == str(text_path)
    )


def load_processed_record(text_path: Path, meta_path: Path) -> dict:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    text = text_path.read_text(encoding="utf-8").strip()

    return {
        "id": meta["id"],
        "source_file": meta["source_file"],
        "source_path": meta["source_path"],
        "text_path": meta["text_path"],
        "modality": meta["modality"],
        "language": meta["language"],
        "sha256": meta["source_sha256"],
        "created_at": meta["created_at"],
        "text": text,
    }


def build_done_metadata(
    source_path: Path,
    text_path: Path,
    modality: str,
    language: str,
) -> dict:
    source_hash = sha256_file(source_path)

    return {
        "id": f"{modality}:{source_hash[:16]}",
        "status": "done",
        "source_file": source_path.name,
        "source_path": str(source_path),
        "text_path": str(text_path),
        "modality": modality,
        "language": language,
        "source_sha256": source_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def record_from_meta_and_text(meta: dict, text: str) -> dict:
    return {
        "id": meta["id"],
        "source_file": meta["source_file"],
        "source_path": meta["source_path"],
        "text_path": meta["text_path"],
        "modality": meta["modality"],
        "language": meta["language"],
        "sha256": meta["source_sha256"],
        "created_at": meta["created_at"],
        "text": text,
    }


# ---------------------------------------------------------------------
# OCR images
# ---------------------------------------------------------------------

def preprocess_image_for_ocr(image: Image.Image) -> Image.Image:
    """
    Prétraitement simple pour améliorer l'OCR :
    - passage en niveaux de gris ;
    - augmentation du contraste ;
    - agrandissement si l'image est petite ;
    - léger renforcement de netteté.
    """
    image = image.convert("RGB")
    gray = ImageOps.grayscale(image)
    gray = ImageOps.autocontrast(gray)

    width, height = gray.size
    max_side = max(width, height)

    if max_side < 1800:
        scale = 1800 / max_side
        new_size = (int(width * scale), int(height * scale))
        gray = gray.resize(new_size, Image.Resampling.LANCZOS)

    gray = gray.filter(ImageFilter.SHARPEN)
    return gray


def ocr_single_image(image: Image.Image, lang: str = "fra") -> str:
    """
    Extrait le texte d'une image avec Tesseract.
    lang='fra' nécessite le paquet système tesseract-ocr-fra.
    """
    processed = preprocess_image_for_ocr(image)

    config = "--oem 3 --psm 3"

    data = pytesseract.image_to_data(
        processed,
        lang=lang,
        config=config,
        output_type=pytesseract.Output.DICT
    )

    lines = {}

    for i, word in enumerate(data["text"]):
        word = word.strip()

        if not word:
            continue

        try:
            confidence = float(data["conf"][i])
        except ValueError:
            confidence = -1

        if confidence < 25:
            continue

        key = (
            data["block_num"][i],
            data["par_num"][i],
            data["line_num"][i],
        )

        lines.setdefault(key, []).append((data["left"][i], word))

    ordered_lines = []
    previous_block = None

    for key in sorted(lines.keys()):
        block_num = key[0]

        if previous_block is not None and block_num != previous_block:
            ordered_lines.append("")

        words = [word for _, word in sorted(lines[key], key=lambda x: x[0])]
        ordered_lines.append(" ".join(words))

        previous_block = block_num

    text = "\n".join(ordered_lines).strip()

    if not text:
        text = pytesseract.image_to_string(
            processed,
            lang=lang,
            config=config
        ).strip()

    return text


def ocr_image_file(image_path: Path, lang: str = "fra") -> str:
    """
    Gère aussi les images multipages, par exemple certains TIFF.
    """
    with Image.open(image_path) as img:
        page_texts = []

        for page_index, frame in enumerate(ImageSequence.Iterator(img), start=1):
            text = ocr_single_image(frame, lang=lang)

            if text:
                if getattr(img, "n_frames", 1) > 1:
                    page_texts.append(f"--- Page {page_index} ---\n{text}")
                else:
                    page_texts.append(text)

    return "\n\n".join(page_texts).strip()


def image_worker(task: dict) -> dict:
    source_path = Path(task["source_path"])
    text_path = Path(task["text_path"])
    meta_path = Path(task["meta_path"])
    tesseract_lang = task["tesseract_lang"]
    language = task["language"]

    text = ocr_image_file(source_path, lang=tesseract_lang)
    atomic_write_text(text_path, text)

    meta = build_done_metadata(
        source_path=source_path,
        text_path=text_path,
        modality="image",
        language=language,
    )

    atomic_write_json(meta_path, meta)

    return record_from_meta_and_text(meta, text)


# ---------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------

def pil_image_from_pdf_page(page, dpi: int = 200) -> Image.Image:
    """
    Rend une page PDF en image PIL pour OCR.
    Utilise PyMuPDF, donc pas besoin de poppler/pdf2image.
    """
    import fitz  # PyMuPDF

    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)

    image = Image.frombytes(
        "RGB",
        [pix.width, pix.height],
        pix.samples
    )

    return image


def extract_pdf_file(
    pdf_path: Path,
    tesseract_lang: str = "fra",
    ocr_mode: str = "auto",
    min_text_chars_per_page: int = 80,
    dpi: int = 200,
) -> str:
    """
    Extrait le texte d'un PDF.

    ocr_mode :
    - "auto"   : OCR seulement si la page contient peu ou pas de texte natif ;
    - "always" : OCR de toutes les pages ;
    - "never"  : extraction texte uniquement, pas d'OCR.
    """
    import fitz  # PyMuPDF

    if ocr_mode not in {"auto", "always", "never"}:
        raise ValueError("ocr_mode doit être 'auto', 'always' ou 'never'.")

    page_texts = []

    with fitz.open(pdf_path) as doc:
        for page_index, page in enumerate(doc, start=1):
            native_text = page.get_text("text").strip()

            should_ocr = (
                ocr_mode == "always"
                or (
                    ocr_mode == "auto"
                    and len(native_text) < min_text_chars_per_page
                )
            )

            if should_ocr and ocr_mode != "never":
                try:
                    image = pil_image_from_pdf_page(page, dpi=dpi)
                    ocr_text = ocr_single_image(image, lang=tesseract_lang).strip()

                    if ocr_text:
                        page_text = ocr_text
                    else:
                        page_text = native_text

                except Exception as e:
                    page_text = native_text
                    print(f"OCR impossible sur {pdf_path.name}, page {page_index} : {e}")
            else:
                page_text = native_text

            if page_text:
                page_texts.append(
                    f"--- Page {page_index} ---\n{page_text}"
                )

    return "\n\n".join(page_texts).strip()


def pdf_worker(task: dict) -> dict:
    source_path = Path(task["source_path"])
    text_path = Path(task["text_path"])
    meta_path = Path(task["meta_path"])

    tesseract_lang = task["tesseract_lang"]
    language = task["language"]
    pdf_ocr_mode = task["pdf_ocr_mode"]
    pdf_min_text_chars = task["pdf_min_text_chars"]
    pdf_dpi = task["pdf_dpi"]

    text = extract_pdf_file(
        pdf_path=source_path,
        tesseract_lang=tesseract_lang,
        ocr_mode=pdf_ocr_mode,
        min_text_chars_per_page=pdf_min_text_chars,
        dpi=pdf_dpi,
    )

    atomic_write_text(text_path, text)

    meta = build_done_metadata(
        source_path=source_path,
        text_path=text_path,
        modality="pdf",
        language=language,
    )

    atomic_write_json(meta_path, meta)

    return record_from_meta_and_text(meta, text)


# ---------------------------------------------------------------------
# Whisper vidéos
# ---------------------------------------------------------------------

def resolve_device(device: str) -> str:
    if device != "auto":
        return device

    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def init_whisper_worker(model_name: str, device: str):
    global _WHISPER_MODEL

    import whisper

    resolved_device = resolve_device(device)
    _WHISPER_MODEL = whisper.load_model(model_name, device=resolved_device)


def video_worker(task: dict) -> dict:
    global _WHISPER_MODEL

    if _WHISPER_MODEL is None:
        raise RuntimeError("Le modèle Whisper n'a pas été initialisé dans le worker.")

    source_path = Path(task["source_path"])
    text_path = Path(task["text_path"])
    meta_path = Path(task["meta_path"])
    language = task["language"]
    device = resolve_device(task["device"])

    result = _WHISPER_MODEL.transcribe(
        str(source_path),
        language=language,
        fp16=(device == "cuda")
    )

    text = result["text"].strip()
    atomic_write_text(text_path, text)

    meta = build_done_metadata(
        source_path=source_path,
        text_path=text_path,
        modality="video",
        language=language,
    )

    atomic_write_json(meta_path, meta)

    return record_from_meta_and_text(meta, text)


# ---------------------------------------------------------------------
# Traitement parallèle
# ---------------------------------------------------------------------

def prepare_tasks(
    files: list[Path],
    output_dir: Path,
    modality: str,
    language: str,
    overwrite: bool,
) -> tuple[list[dict], list[dict]]:
    records = []
    tasks = []

    text_output_dir = output_dir / "texts" / f"{modality}s"
    metadata_output_dir = output_dir / "metadata" / f"{modality}s"

    for source_path in tqdm(files, desc=f"Scan {modality}s"):
        stem = safe_filename_stem(source_path)
        text_path = text_output_dir / f"{stem}.txt"
        meta_path = metadata_output_dir / f"{stem}.meta.json"

        if not overwrite and is_already_processed(source_path, text_path, meta_path):
            records.append(load_processed_record(text_path, meta_path))
        else:
            tasks.append({
                "source_path": str(source_path),
                "text_path": str(text_path),
                "meta_path": str(meta_path),
                "language": language,
            })

    return records, tasks


def run_parallel_tasks(
    tasks: list[dict],
    worker_fn,
    max_workers: int,
    desc: str,
    log_path: Path,
    initializer=None,
    initargs=None,
) -> list[dict]:
    records = []

    if not tasks:
        print(f"{desc} : rien à traiter.")
        return records

    max_workers = max(1, max_workers)

    print(f"{desc} : {len(tasks)} fichier(s) à traiter avec {max_workers} worker(s).")

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=initializer,
        initargs=initargs or (),
    ) as executor:
        futures = {
            executor.submit(worker_fn, task): task
            for task in tasks
        }

        try:
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=desc
            ):
                task = futures[future]

                try:
                    record = future.result()
                    records.append(record)

                    write_event(
                        log_path,
                        "done",
                        {
                            "source_path": task["source_path"],
                            "text_path": task["text_path"],
                        }
                    )

                except Exception as e:
                    write_event(
                        log_path,
                        "error",
                        {
                            "source_path": task["source_path"],
                            "error": str(e),
                        }
                    )
                    print(f"Erreur sur {Path(task['source_path']).name} : {e}")

        except KeyboardInterrupt:
            print("\nInterruption détectée. Les fichiers déjà terminés ne seront pas retraités.")
            executor.shutdown(wait=False, cancel_futures=True)
            raise

    return records


def process_images(
    images_dir: Path,
    output_dir: Path,
    tesseract_lang: str,
    language: str,
    image_workers: int,
    overwrite: bool,
    log_path: Path,
) -> list[dict]:
    images = iter_files(images_dir, IMAGE_EXTENSIONS)

    if not images:
        print(f"Aucune image trouvée dans : {images_dir}")
        return []

    already_done, tasks = prepare_tasks(
        files=images,
        output_dir=output_dir,
        modality="image",
        language=language,
        overwrite=overwrite,
    )

    for task in tasks:
        task["tesseract_lang"] = tesseract_lang

    processed = run_parallel_tasks(
        tasks=tasks,
        worker_fn=image_worker,
        max_workers=image_workers,
        desc="OCR images",
        log_path=log_path,
    )

    return already_done + processed


def process_pdfs(
    pdfs_dir: Path,
    output_dir: Path,
    tesseract_lang: str,
    language: str,
    pdf_workers: int,
    pdf_ocr_mode: str,
    pdf_min_text_chars: int,
    pdf_dpi: int,
    overwrite: bool,
    log_path: Path,
) -> list[dict]:
    pdfs = iter_files(pdfs_dir, PDF_EXTENSIONS)

    if not pdfs:
        print(f"Aucun PDF trouvé dans : {pdfs_dir}")
        return []

    already_done, tasks = prepare_tasks(
        files=pdfs,
        output_dir=output_dir,
        modality="pdf",
        language=language,
        overwrite=overwrite,
    )

    for task in tasks:
        task["tesseract_lang"] = tesseract_lang
        task["pdf_ocr_mode"] = pdf_ocr_mode
        task["pdf_min_text_chars"] = pdf_min_text_chars
        task["pdf_dpi"] = pdf_dpi

    processed = run_parallel_tasks(
        tasks=tasks,
        worker_fn=pdf_worker,
        max_workers=pdf_workers,
        desc="Extraction PDFs",
        log_path=log_path,
    )

    return already_done + processed


def process_videos(
    videos_dir: Path,
    output_dir: Path,
    whisper_model_name: str,
    language: str,
    video_workers: int,
    device: str,
    overwrite: bool,
    log_path: Path,
) -> list[dict]:
    videos = iter_files(videos_dir, VIDEO_EXTENSIONS)

    if not videos:
        print(f"Aucune vidéo trouvée dans : {videos_dir}")
        return []

    already_done, tasks = prepare_tasks(
        files=videos,
        output_dir=output_dir,
        modality="video",
        language=language,
        overwrite=overwrite,
    )

    for task in tasks:
        task["device"] = device

    processed = run_parallel_tasks(
        tasks=tasks,
        worker_fn=video_worker,
        max_workers=video_workers,
        desc="Transcription vidéos",
        log_path=log_path,
        initializer=init_whisper_worker,
        initargs=(whisper_model_name, device),
    )

    return already_done + processed


# ---------------------------------------------------------------------
# Chunking RAG
# ---------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    """
    Découpage simple pour RAG.
    Les paragraphes sont conservés autant que possible.
    """
    text = re.sub(r"\n{3,}", "\n\n", text.strip())

    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            if current:
                chunks.append(current.strip())
                current = ""

            start = 0

            while start < len(paragraph):
                end = start + chunk_size
                chunks.append(paragraph[start:end].strip())

                next_start = end - overlap
                if next_start <= start:
                    next_start = start + 1
                start = next_start

            continue

        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph

        if len(candidate) <= chunk_size:
            current = candidate
        else:
            chunks.append(current.strip())
            current = paragraph

    if current:
        chunks.append(current.strip())

    return chunks


def build_chunks(records: list[dict], chunk_size: int, chunk_overlap: int) -> list[dict]:
    chunk_records = []

    for record in tqdm(records, desc="Création des chunks"):
        chunks = chunk_text(
            record["text"],
            chunk_size=chunk_size,
            overlap=chunk_overlap,
        )

        for i, chunk in enumerate(chunks):
            chunk_records.append({
                "id": f"{record['id']}:chunk:{i}",
                "document_id": record["id"],
                "chunk_index": i,
                "source_file": record["source_file"],
                "source_path": record["source_path"],
                "text_path": record["text_path"],
                "modality": record["modality"],
                "language": record["language"],
                "text": chunk,
            })

    return chunk_records


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    default_parallel_workers = max(1, min(cpu_count() - 1, 4))

    parser = argparse.ArgumentParser(
        description="Prépare un corpus RAG depuis des vidéos, des images et des PDFs."
    )

    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=None,
        help="Dossier contenant les vidéos à transcrire."
    )

    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Dossier contenant les images à lire par OCR."
    )

    parser.add_argument(
        "--pdfs-dir",
        type=Path,
        default=None,
        help="Dossier contenant les PDFs à traiter."
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Dossier de sortie pour les textes, métadonnées et JSONL."
    )

    parser.add_argument(
        "--whisper-model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Modèle Whisper à utiliser."
    )

    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device Whisper."
    )

    parser.add_argument(
        "--language",
        default="fr",
        help="Langue principale du corpus."
    )

    parser.add_argument(
        "--tesseract-lang",
        default="fra",
        help="Langue Tesseract pour l'OCR. Pour le français : fra."
    )

    parser.add_argument(
        "--video-workers",
        type=int,
        default=1,
        help="Nombre de workers pour Whisper. Garder 1 sur GPU sauf si beaucoup de VRAM."
    )

    parser.add_argument(
        "--image-workers",
        type=int,
        default=default_parallel_workers,
        help="Nombre de workers pour l'OCR images."
    )

    parser.add_argument(
        "--pdf-workers",
        type=int,
        default=default_parallel_workers,
        help="Nombre de workers pour l'extraction des PDFs."
    )

    parser.add_argument(
        "--pdf-ocr-mode",
        default="auto",
        choices=["auto", "always", "never"],
        help="Mode OCR pour les PDFs : auto, always ou never."
    )

    parser.add_argument(
        "--pdf-min-text-chars",
        type=int,
        default=80,
        help="Nombre minimal de caractères natifs par page sous lequel l'OCR est déclenché en mode auto."
    )

    parser.add_argument(
        "--pdf-dpi",
        type=int,
        default=200,
        help="Résolution utilisée pour rendre les pages PDF avant OCR."
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1200,
        help="Taille maximale des chunks."
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Recouvrement entre chunks."
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force le retraitement des fichiers déjà terminés."
    )

    args = parser.parse_args()

    if args.videos_dir is None and args.images_dir is None and args.pdfs_dir is None:
        raise ValueError(
            "Il faut fournir --videos-dir, --images-dir, --pdfs-dir, ou plusieurs d'entre eux."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    log_path = args.output_dir / "processing_log.jsonl"

    all_records = []

    write_event(
        log_path,
        "start",
        {
            "videos_dir": str(args.videos_dir) if args.videos_dir else None,
            "images_dir": str(args.images_dir) if args.images_dir else None,
            "pdfs_dir": str(args.pdfs_dir) if args.pdfs_dir else None,
            "output_dir": str(args.output_dir),
            "video_workers": args.video_workers,
            "image_workers": args.image_workers,
            "pdf_workers": args.pdf_workers,
            "whisper_model": args.whisper_model,
            "device": args.device,
            "pdf_ocr_mode": args.pdf_ocr_mode,
            "pdf_dpi": args.pdf_dpi,
        }
    )

    try:
        if args.images_dir is not None:
            image_records = process_images(
                images_dir=args.images_dir,
                output_dir=args.output_dir,
                tesseract_lang=args.tesseract_lang,
                language=args.language,
                image_workers=args.image_workers,
                overwrite=args.overwrite,
                log_path=log_path,
            )
            all_records.extend(image_records)

        if args.pdfs_dir is not None:
            pdf_records = process_pdfs(
                pdfs_dir=args.pdfs_dir,
                output_dir=args.output_dir,
                tesseract_lang=args.tesseract_lang,
                language=args.language,
                pdf_workers=args.pdf_workers,
                pdf_ocr_mode=args.pdf_ocr_mode,
                pdf_min_text_chars=args.pdf_min_text_chars,
                pdf_dpi=args.pdf_dpi,
                overwrite=args.overwrite,
                log_path=log_path,
            )
            all_records.extend(pdf_records)

        if args.videos_dir is not None:
            video_records = process_videos(
                videos_dir=args.videos_dir,
                output_dir=args.output_dir,
                whisper_model_name=args.whisper_model,
                language=args.language,
                video_workers=args.video_workers,
                device=args.device,
                overwrite=args.overwrite,
                log_path=log_path,
            )
            all_records.extend(video_records)

        documents_path = args.output_dir / "documents.jsonl"
        chunks_path = args.output_dir / "chunks.jsonl"

        write_jsonl(documents_path, all_records)

        chunk_records = build_chunks(
            records=all_records,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )

        write_jsonl(chunks_path, chunk_records)

        write_event(
            log_path,
            "finish",
            {
                "documents_count": len(all_records),
                "chunks_count": len(chunk_records),
                "documents_path": str(documents_path),
                "chunks_path": str(chunks_path),
            }
        )

        print("\nPréparation terminée.")
        print(f"Documents préparés : {len(all_records)}")
        print(f"Chunks générés : {len(chunk_records)}")
        print(f"Fichiers texte : {args.output_dir / 'texts'}")
        print(f"Métadonnées : {args.output_dir / 'metadata'}")
        print(f"Documents JSONL : {documents_path}")
        print(f"Chunks JSONL : {chunks_path}")
        print(f"Log : {log_path}")

    except KeyboardInterrupt:
        write_event(
            log_path,
            "interrupted",
            {
                "message": "Interruption utilisateur. Relancer la même commande pour reprendre."
            }
        )
        print("\nExécution interrompue.")
        print("Relance la même commande : seuls les fichiers non terminés seront repris.")
        raise


if __name__ == "__main__":
    main()
