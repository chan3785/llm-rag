"""Embeds a folder of images with CLIP and upserts them into Qdrant.

Each image may have a sidecar text file with the same name (cat.jpg + cat.txt);
its content is stored as the `description` given to the LLM at chat time.

Usage: uv run python -m scripts.ingest <image-folder> [--recreate]
"""

import argparse
import uuid
from pathlib import Path

from PIL import Image
from qdrant_client import QdrantClient, models

from app.config import settings
from app.embedding import ClipEmbedder

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
BATCH_SIZE = 32


def ensure_collection(client: QdrantClient, name: str, dimension: int, recreate: bool) -> None:
    if recreate and client.collection_exists(name):
        client.delete_collection(name)
    if not client.collection_exists(name):
        client.create_collection(
            name,
            vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
        )
        return
    existing = client.get_collection(name).config.params.vectors.size
    if existing != dimension:
        raise SystemExit(
            f"Collection '{name}' has vector size {existing}, but {settings.clip_model} "
            f"produces {dimension}. Re-run with --recreate."
        )


def point_id(path: Path) -> str:
    # Stable id per file, so re-running the script updates instead of duplicating.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, str(path.resolve())))


def read_description(path: Path) -> str:
    sidecar = path.with_suffix(".txt")
    return sidecar.read_text(encoding="utf-8").strip() if sidecar.exists() else ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--recreate", action="store_true", help="drop and recreate the collection")
    args = parser.parse_args()

    paths = sorted(p for p in args.folder.rglob("*") if p.suffix.lower() in EXTENSIONS)
    if not paths:
        raise SystemExit(f"No images found in {args.folder}")

    clip = ClipEmbedder(settings.clip_model)
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    ensure_collection(client, settings.qdrant_collection, clip.dimension, args.recreate)

    for start in range(0, len(paths), BATCH_SIZE):
        batch = paths[start : start + BATCH_SIZE]
        vectors = clip.embed_images([Image.open(p).convert("RGB") for p in batch])
        client.upsert(
            settings.qdrant_collection,
            points=[
                models.PointStruct(
                    id=point_id(p),
                    vector=vector,
                    payload={"path": str(p), "description": read_description(p)},
                )
                for p, vector in zip(batch, vectors)
            ],
        )
        print(f"{start + len(batch)}/{len(paths)} images ingested")

    print(f"Done: collection '{settings.qdrant_collection}' at {settings.qdrant_url}")


if __name__ == "__main__":
    main()
