"""Unit tests for the content-addressed embedding cache in src/retrieval/embedding_cache.py."""

import numpy as np

from src.retrieval import embedding_cache


class CountingEmbedder:
    """Deterministic fake embedder that records which texts it was asked to embed."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def __call__(self, texts: list[str]) -> np.ndarray:
        self.calls.append(list(texts))
        return np.array([[float(len(t)), float(sum(map(ord, t)))] for t in texts], np.float32)


def test_first_build_embeds_everything(tmp_path):
    emb = CountingEmbedder()
    vectors, n_new = embedding_cache.get_or_embed(tmp_path / "c.npy", ["a", "bb"], emb)
    assert n_new == 2 and vectors.shape == (2, 2)
    assert emb.calls == [["a", "bb"]]


def test_rebuild_embeds_only_new_texts_and_keeps_order(tmp_path):
    path = tmp_path / "c.npy"
    embedding_cache.get_or_embed(path, ["a", "bb", "ccc"], CountingEmbedder())
    emb = CountingEmbedder()
    # Reordered, one text dropped, one added: only "dddd" is new.
    vectors, n_new = embedding_cache.get_or_embed(path, ["ccc", "dddd", "a"], emb)
    assert n_new == 1 and emb.calls == [["dddd"]]
    expected = CountingEmbedder()(["ccc", "dddd", "a"])
    assert np.array_equal(vectors, expected)


def test_same_count_but_changed_text_is_not_reused(tmp_path):
    # The old row-count check would have reused row 1 for "xx" here.
    path = tmp_path / "c.npy"
    embedding_cache.get_or_embed(path, ["a", "bb"], CountingEmbedder())
    emb = CountingEmbedder()
    vectors, n_new = embedding_cache.get_or_embed(path, ["a", "xx"], emb)
    assert n_new == 1 and emb.calls == [["xx"]]
    assert np.array_equal(vectors[1], CountingEmbedder()(["xx"])[0])


def test_duplicate_new_texts_are_embedded_once(tmp_path):
    emb = CountingEmbedder()
    vectors, n_new = embedding_cache.get_or_embed(tmp_path / "c.npy", ["z", "z", "y"], emb)
    assert n_new == 2 and emb.calls == [["z", "y"]]
    assert np.array_equal(vectors[0], vectors[1])


def test_npy_without_keys_file_is_not_trusted(tmp_path):
    path = tmp_path / "c.npy"
    np.save(path, np.zeros((2, 2), np.float32))
    emb = CountingEmbedder()
    _, n_new = embedding_cache.get_or_embed(path, ["a", "bb"], emb)
    assert n_new == 2
