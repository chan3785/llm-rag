"""Measures landmark classification on held-out images that were NOT ingested.

Layout matches ingestion (one folder per landmark). Put photos of places that are not
registered in a folder named `_unknown`; they count as correct when rejected.

    data/eval/
    ├── 경복궁/*.jpg
    ├── 불국사/*.jpg
    └── _unknown/*.jpg

Usage: uv run python -m scripts.evaluate <eval-folder>
"""

import argparse
from pathlib import Path

from PIL import Image
from qdrant_client import QdrantClient, models

from app.classify import Classification, vote
from app.config import settings
from app.embedding import ClipEmbedder

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
UNKNOWN_FOLDER = "_unknown"
SWEEP = [round(0.5 + 0.05 * i, 2) for i in range(10)]


def expected_label(path: Path) -> str | None:
    return None if path.parent.name == UNKNOWN_FOLDER else path.parent.name


def score(cases: list[tuple[str | None, Classification]]) -> dict[str, float]:
    known = [(y, c) for y, c in cases if y is not None]
    unknown = [(y, c) for y, c in cases if y is None]
    ratio = lambda hits, items: hits / len(items) if items else float("nan")  # noqa: E731
    return {
        "overall": ratio(sum(y == c.landmark for y, c in cases), cases),
        "known_correct": ratio(sum(y == c.landmark for y, c in known), known),
        "known_rejected": ratio(sum(c.landmark is None for _, c in known), known),
        "unknown_rejected": ratio(sum(c.landmark is None for _, c in unknown), unknown),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()

    paths = sorted(p for p in args.folder.rglob("*") if p.suffix.lower() in EXTENSIONS)
    if not paths:
        raise SystemExit(f"No images found in {args.folder}")

    clip = ClipEmbedder(settings.clip_model)
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    vectors = clip.embed_images([Image.open(p).convert("RGB") for p in paths])
    responses = client.query_batch_points(
        settings.qdrant_collection,
        requests=[
            models.QueryRequest(query=v, limit=settings.classify_top_k, with_payload=True)
            for v in vectors
        ],
    )
    neighbours = [r.points for r in responses]

    threshold = settings.classify_threshold
    print(f"Per image (top_k={settings.classify_top_k}, threshold={threshold}):")
    for path, points in zip(paths, neighbours):
        c = vote(points, threshold, settings.classify_per_landmark)
        expected = expected_label(path)
        mark = "✓" if c.landmark == expected else "✗"
        print(
            f"  {mark} {path.parent.name}/{path.name}: "
            f"{c.landmark or 'unknown'} (score {c.score:.2f}, {c.votes}/{c.total} votes)"
        )

    print("\nThreshold sweep:")
    print("  threshold  overall  known_correct  known_rejected  unknown_rejected")
    for t in sorted({*SWEEP, threshold}):
        cases = [
            (expected_label(p), vote(pts, t, settings.classify_per_landmark))
            for p, pts in zip(paths, neighbours)
        ]
        s = score(cases)
        marker = "  <- current" if t == threshold else ""
        print(
            f"  {t:9.2f}  {s['overall']:7.0%}  {s['known_correct']:13.0%}  "
            f"{s['known_rejected']:14.0%}  {s['unknown_rejected']:16.0%}{marker}"
        )


if __name__ == "__main__":
    main()
