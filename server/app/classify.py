from collections import defaultdict
from dataclasses import dataclass

from qdrant_client.models import ScoredPoint


@dataclass(frozen=True)
class Classification:
    landmark: str | None  # accepted landmark; None when the candidate is below the threshold
    candidate: str | None  # closest landmark, even when rejected
    score: float  # highest similarity among the candidate's images
    votes: int  # how many of the retrieved images belong to the candidate
    total: int  # how many labelled images were retrieved
    description: str


def vote(points: list[ScoredPoint], threshold: float, per_landmark: int = 3) -> Classification:
    """Classifies by the mean similarity of each landmark's `per_landmark` closest images.

    Averaging a fixed number per landmark (instead of summing every hit) keeps landmarks with
    many reference photos from winning just by volume. The candidate is rejected as unknown when
    even its closest image is below `threshold`.
    """
    scores: dict[str, list[float]] = defaultdict(list)
    descriptions: dict[str, str] = {}
    for point in points:  # Qdrant returns points sorted by score, highest first
        payload = point.payload or {}
        label = payload.get("landmark")
        if not label:
            continue
        scores[label].append(point.score)
        descriptions.setdefault(label, payload.get("description", ""))

    if not scores:
        return Classification(None, None, 0.0, 0, 0, "")

    def landmark_score(label: str) -> float:
        top = scores[label][:per_landmark]
        return sum(top) / len(top)

    candidate = max(scores, key=landmark_score)
    best = scores[candidate][0]
    votes = len(scores[candidate])
    total = sum(len(s) for s in scores.values())
    if best < threshold:
        return Classification(None, candidate, best, votes, total, "")
    return Classification(candidate, candidate, best, votes, total, descriptions[candidate])
