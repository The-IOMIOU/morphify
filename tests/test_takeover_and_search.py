"""Unit cover for the pieces that do not need models or a network.

The model-driven paths (parsing, matting, the swap itself) are exercised by
the scripts under setup/; these are the bits of logic that are easy to break
silently in a refactor.
"""

import numpy as np
import pytest

from modules import takeover
from modules.face_identity import _reject_outliers, find_group, group_key
from modules.image_search import relax_query, slugify


# ─── image search ────────────────────────────────────────────────────────


@pytest.mark.parametrize("query,expected", [
    ("kai cenat portrait", "kai cenat"),
    ("lebron james close up", "lebron james"),
    ("messi headshot photo", "messi"),
    ("someone", "someone"),
])
def test_relax_query_strips_photo_words(query, expected):
    assert relax_query(query) == expected


def test_relax_query_keeps_something_when_all_words_are_generic():
    # "portrait photo" is all qualifiers; returning "" would search for
    # nothing at all, which is worse than searching literally.
    assert relax_query("portrait photo") == "portrait photo"


def test_slugify_makes_a_searchable_filename():
    assert slugify("LeBron James close up") == "lebron-james-close-up"
    assert slugify("!!!") == "face"
    assert slugify("") == "face"
    assert "/" not in slugify("a/b\\c")


# ─── identity grouping ───────────────────────────────────────────────────


@pytest.mark.parametrize("name,expected", [
    ("kai-cenat-01.jpg", "kai-cenat"),
    ("kai-cenat-12.png", "kai-cenat"),
    ("kai_cenat_03.jpg", "kai_cenat"),
    ("holiday.png", "holiday"),
])
def test_group_key(tmp_path, name, expected):
    assert group_key(str(tmp_path / name)) == expected


def test_find_group_collects_the_same_person(tmp_path):
    for name in ("kai-cenat-01.jpg", "kai-cenat-02.jpg", "someone-else-01.jpg"):
        (tmp_path / name).write_bytes(b"x")
    group = find_group(str(tmp_path / "kai-cenat-01.jpg"))
    assert len(group) == 2
    assert all("kai-cenat" in g for g in group)


def _unit(vector):
    vector = np.asarray(vector, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def test_outlier_rejection_drops_a_different_person():
    # Four embeddings clustered together, one pointing elsewhere.
    base = _unit([1.0, 0.0, 0.0, 0.0])
    group = [_unit(base + np.random.RandomState(i).normal(0, 0.05, 4))
             for i in range(4)]
    impostor = _unit([0.0, 1.0, 0.0, 0.0])
    kept, dropped = _reject_outliers(group + [impostor],
                                     [f"g{i}" for i in range(4)] + ["impostor"])
    assert dropped == ["impostor"]
    assert len(kept) == 4


def test_outlier_rejection_keeps_small_groups_intact():
    # With fewer than three there is no majority to appeal to.
    a, b = _unit([1, 0, 0, 0]), _unit([0, 1, 0, 0])
    kept, dropped = _reject_outliers([a, b], ["a", "b"])
    assert len(kept) == 2 and dropped == []


def test_outlier_rejection_never_empties_the_group():
    # Mutually dissimilar embeddings must not all be thrown away.
    vectors = [_unit(v) for v in ([1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0])]
    kept, _dropped = _reject_outliers(vectors, ["a", "b", "c"])
    assert kept


# ─── takeover geometry and blending ──────────────────────────────────────


def test_alpha_bounds_finds_the_subject():
    alpha = np.zeros((100, 200), dtype=np.float32)
    alpha[30:60, 80:120] = 1.0
    x1, y1, x2, y2 = takeover._alpha_bounds(alpha, pad=0)
    assert (x1, y1, x2, y2) == (80, 30, 120, 60)


def test_alpha_bounds_pads_but_stays_inside():
    alpha = np.zeros((50, 50), dtype=np.float32)
    alpha[0:5, 0:5] = 1.0
    x1, y1, x2, y2 = takeover._alpha_bounds(alpha, pad=20)
    assert (x1, y1) == (0, 0)
    assert x2 <= 50 and y2 <= 50


def test_alpha_bounds_returns_none_when_empty():
    assert takeover._alpha_bounds(np.zeros((10, 10), dtype=np.float32)) is None


def test_blend_respects_alpha_extremes():
    base = np.zeros((8, 8, 3), dtype=np.uint8)
    overlay = np.full((8, 8, 3), 200, dtype=np.uint8)
    assert takeover.blend(base, overlay, np.zeros((8, 8), np.float32)).max() == 0
    assert takeover.blend(base, overlay, np.ones((8, 8), np.float32)).min() == 200


def test_blend_is_linear_at_half():
    base = np.zeros((4, 4, 3), dtype=np.uint8)
    overlay = np.full((4, 4, 3), 100, dtype=np.uint8)
    out = takeover.blend(base, overlay, np.full((4, 4), 0.5, np.float32))
    assert abs(int(out[0, 0, 0]) - 50) <= 1


def test_cover_fills_the_target_exactly():
    image = np.zeros((100, 50, 3), dtype=np.uint8)
    out = takeover._cover(image, 200, 100)
    assert out.shape[:2] == (100, 200)


def test_mask_from_labels_selects_requested_classes():
    labels = np.array([[0, 1], [17, 14]], dtype=np.uint8)
    mask = takeover.mask_from_labels(labels, (takeover.HAIR,))
    assert mask.tolist() == [[False, False], [True, False]]


def test_transfer_tone_is_a_noop_at_zero_strength():
    image = np.full((16, 16, 3), 120, dtype=np.uint8)
    region = np.ones((16, 16), dtype=np.float32)
    stats = (np.array([50.0, 128.0, 128.0]), np.array([10.0, 5.0, 5.0]))
    assert np.array_equal(
        takeover.transfer_tone(image, region, stats, strength=0.0), image)


def test_transfer_tone_moves_toward_the_target():
    rng = np.random.RandomState(0)
    image = rng.randint(40, 80, (32, 32, 3)).astype(np.uint8)
    region = np.ones((32, 32), dtype=np.float32)
    bright = (np.array([200.0, 128.0, 128.0]), np.array([8.0, 4.0, 4.0]))
    out = takeover.transfer_tone(image, region, bright, strength=1.0)
    assert out.mean() > image.mean()
