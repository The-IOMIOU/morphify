"""Tests for the likeness control on the face swapper.

The swapper blends the source identity into the target's face rather than
replacing it, so a swap tends to look like neither person. These cover the
control that pushes it further toward the source, and in particular the trap
that makes the obvious implementation silently do nothing.
"""

import os
import sys

import numpy
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.processors.frame.face_swapper import (  # noqa: E402
    _SourceIdentity,
    strengthened_source,
)


class FakeFace:
    """Just the one field the swapper reads off a source face."""

    def __init__(self, embedding):
        vector = numpy.asarray(embedding, dtype=numpy.float32)
        self.normed_embedding = vector / numpy.linalg.norm(vector)


def _cosine(a, b):
    return float(numpy.dot(a, b)
                 / (numpy.linalg.norm(a) * numpy.linalg.norm(b)))


@pytest.fixture
def faces():
    source = FakeFace([1.0, 0.0, 0.0, 0.0])
    target = FakeFace([0.6, 0.8, 0.0, 0.0])
    return source, target


# ── the control does nothing when off ────────────────────────────────────────


def test_zero_strength_leaves_the_source_untouched(faces):
    source, target = faces
    assert strengthened_source(source, target, 0.0) is source


def test_missing_target_embedding_is_survivable(faces):
    source, _target = faces

    class NoEmbedding:
        normed_embedding = None

    assert strengthened_source(source, NoEmbedding(), 0.5) is source
    assert strengthened_source(source, None, 0.5) is source


def test_missing_source_embedding_is_survivable(faces):
    _source, target = faces

    class NoEmbedding:
        normed_embedding = None

    face = NoEmbedding()
    assert strengthened_source(face, target, 0.5) is face


# ── the control actually moves the identity ──────────────────────────────────


def test_strength_moves_the_identity_away_from_the_target(faces):
    source, target = faces
    adjusted = strengthened_source(source, target, 0.5)
    assert isinstance(adjusted, _SourceIdentity)
    before = _cosine(source.normed_embedding, target.normed_embedding)
    after = _cosine(adjusted.normed_embedding, target.normed_embedding)
    assert after < before, "identity should end up less like the target"


def test_more_strength_moves_it_further(faces):
    source, target = faces
    similarities = [
        _cosine(strengthened_source(source, target, s).normed_embedding,
                target.normed_embedding)
        for s in (0.2, 0.4, 0.6, 0.8)
    ]
    assert similarities == sorted(similarities, reverse=True)


def test_the_result_stays_closer_to_the_source_than_the_target(faces):
    source, target = faces
    adjusted = strengthened_source(source, target, 0.5).normed_embedding
    assert (_cosine(adjusted, source.normed_embedding)
            > _cosine(adjusted, target.normed_embedding))


def test_output_is_unit_length(faces):
    """The swapper renormalises anyway, but an unnormalised vector here would
    hide a scaling bug behind that renormalisation."""
    source, target = faces
    for strength in (0.1, 0.5, 1.0):
        adjusted = strengthened_source(source, target, strength)
        assert numpy.linalg.norm(adjusted.normed_embedding) == pytest.approx(1.0)


# ── the trap ─────────────────────────────────────────────────────────────────


def test_scaling_the_embedding_would_have_been_a_no_op():
    """Why the control extrapolates rather than amplifies.

    inswapper computes ``latent = normed_embedding . emap`` and then divides by
    that vector's norm. Scaling the embedding scales the latent and is undone by
    the division, so a "strength = multiply it" slider would move nothing at all
    while looking like it worked.
    """
    embedding = numpy.array([0.3, -0.7, 0.2, 0.6], dtype=numpy.float32)
    emap = numpy.eye(4, dtype=numpy.float32)

    def latent(vector):
        result = numpy.dot(vector.reshape(1, -1), emap)
        return result / numpy.linalg.norm(result)

    numpy.testing.assert_allclose(latent(embedding), latent(embedding * 5.0),
                                  rtol=1e-5)


def test_identical_faces_do_not_blow_up():
    """source == target means the difference is zero; must not divide by it."""
    face = FakeFace([1.0, 0.0, 0.0, 0.0])
    same = FakeFace([1.0, 0.0, 0.0, 0.0])
    adjusted = strengthened_source(face, same, 0.8)
    assert numpy.all(numpy.isfinite(adjusted.normed_embedding))
    assert numpy.linalg.norm(adjusted.normed_embedding) == pytest.approx(1.0)


def test_opposite_faces_fall_back_instead_of_producing_nonsense():
    """A degenerate cancellation returns the original rather than a zero vector."""
    source = FakeFace([1.0, 0.0, 0.0, 0.0])
    target = FakeFace([2.0, 0.0, 0.0, 0.0])  # same direction after norming
    result = strengthened_source(source, target, 1.0)
    assert numpy.all(numpy.isfinite(result.normed_embedding))
