"""Embeds landmark reference images with CLIP and upserts them into Qdrant.

Layout: one folder per landmark; the folder name is the label.

    data/landmarks/
    ├── 경복궁/
    │   ├── _info.txt   (optional landmark description given to the LLM)
    │   ├── 001.jpg
    │   └── ...
    └── 불국사/...

Usage: uv run python -m scripts.ingest <landmarks-folder> [--recreate]
"""

import argparse
import uuid
from collections import Counter
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
        client.create_payload_index(
            name, field_name="landmark", field_schema=models.PayloadSchemaType.KEYWORD
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


def find_images(root: Path) -> list[Path]:
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in EXTENSIONS)
    misplaced = [p for p in paths if p.parent.resolve() == root.resolve()]
    if misplaced:
        raise SystemExit(
            f"Put images inside a landmark folder (e.g. {root}/경복궁/{misplaced[0].name}); "
            f"{len(misplaced)} image(s) are directly in {root}."
        )
    return paths


def read_landmark(path: Path) -> tuple[str, str]:
    """Returns (label, description) from the image's folder name and its _info.txt."""
    info = path.parent / "_info.txt"
    description = info.read_text(encoding="utf-8").strip() if info.exists() else ""
    return path.parent.name, description


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--recreate", action="store_true", help="drop and recreate the collection")
    args = parser.parse_args()

    paths = find_images(args.folder)
    if not paths:
        raise SystemExit(f"No images found in {args.folder}")

    clip = ClipEmbedder(settings.clip_model)
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    ensure_collection(client, settings.qdrant_collection, clip.dimension, args.recreate)

    for start in range(0, len(paths), BATCH_SIZE):
        batch = paths[start : start + BATCH_SIZE]
        vectors = clip.embed_images([Image.open(p).convert("RGB") for p in batch])
        points = []
        for p, vector in zip(batch, vectors):
            landmark, description = read_landmark(p)
            payload = {"landmark": landmark, "description": description, "path": str(p)}
            points.append(models.PointStruct(id=point_id(p), vector=vector, payload=payload))
        client.upsert(settings.qdrant_collection, points=points)
        print(f"{start + len(batch)}/{len(paths)} images ingested")

    counts = Counter(p.parent.name for p in paths)
    print(f"Done: collection '{settings.qdrant_collection}' at {settings.qdrant_url}")
    for landmark, n in sorted(counts.items()):
        print(f"  {landmark}: {n}")


if __name__ == "__main__":
    main()
