from qdrant_client.models import ScoredPoint

from app.classify import vote


def points(*items: tuple[float, str]) -> list[ScoredPoint]:
    return [
        ScoredPoint(
            id=i,
            version=0,
            score=score,
            payload={"landmark": label, "description": f"{label} info"},
        )
        for i, (score, label) in enumerate(sorted(items, reverse=True))
    ]


def test_picks_landmark_with_highest_mean_of_closest_images():
    # A's closest three average 0.88; B's only match is 0.86.
    result = vote(points((0.9, "A"), (0.88, "A"), (0.86, "A"), (0.86, "B")), threshold=0.8)

    assert result.landmark == "A"
    assert result.score == 0.9
    assert (result.votes, result.total) == (3, 4)
    assert result.description == "A info"


def test_a_weak_extra_match_lowers_a_landmarks_score():
    # A's closest three average 0.79 because of the weak 0.6 match, so B (0.86) wins.
    result = vote(points((0.9, "A"), (0.88, "A"), (0.6, "A"), (0.86, "B")), threshold=0.8)

    assert result.landmark == "B"
    assert result.description == "B info"


def test_more_reference_images_do_not_win_by_volume():
    # B has many mediocre matches; A has fewer but closer ones.
    many_b = [(0.82, "B")] * 8
    result = vote(points((0.9, "A"), (0.89, "A"), (0.88, "A"), *many_b), threshold=0.8)

    assert result.landmark == "A"


def test_rejects_as_unknown_below_threshold():
    result = vote(points((0.75, "A"), (0.7, "B")), threshold=0.8)

    assert result.landmark is None
    assert result.candidate == "A"
    assert result.score == 0.75
    assert result.description == ""


def test_ignores_points_without_landmark_label():
    unlabelled = ScoredPoint(id=99, version=0, score=0.99, payload={"description": "x"})

    result = vote([unlabelled, *points((0.85, "A"))], threshold=0.8)

    assert result.landmark == "A"
    assert result.total == 1


def test_empty_result_is_unknown():
    result = vote([], threshold=0.8)

    assert result.landmark is None
    assert result.candidate is None
    assert result.total == 0
