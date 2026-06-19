from src.data.chunking import chunk_text
from src.data.preprocess import normalize, tokenize


def test_short_text_is_single_chunk():
    text = "вклад это деньги в банке"
    assert chunk_text(text, chunk_size=120, overlap=20) == [text]


def test_long_text_splits_with_overlap():
    words = [f"слово{i}" for i in range(300)]
    chunks = chunk_text(" ".join(words), chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(chunks)


def test_empty_text_gives_no_chunks():
    assert chunk_text("   ") == []


def test_tokenize_drops_stopwords_and_normalizes_yo():
    tokens = tokenize("Ещё раз про ВКЛАДЫ и проценты")
    assert "вклады" in tokens
    assert "и" not in tokens


def test_normalize_collapses_spaces():
    assert normalize("  Привет   мир ") == "привет мир"
