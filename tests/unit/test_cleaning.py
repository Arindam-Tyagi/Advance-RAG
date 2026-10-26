"""Tests for clean_text."""

from rag_app.ingestion.cleaning import (
    build_vocabulary,
    clean_text,
    rejoin_hyphenated_words,
    replace_ligatures,
)


def test_windows_and_old_mac_line_endings_become_newlines():
    assert clean_text("one\r\ntwo\rthree") == "one\ntwo\nthree"


def test_null_characters_are_removed():
    assert clean_text("re\x00trieval") == "retrieval"


def test_trailing_spaces_on_each_line_are_removed():
    assert clean_text("first   \nsecond\t") == "first\nsecond"


def test_many_blank_lines_collapse_to_one_blank_line():
    assert clean_text("para one\n\n\n\n\npara two") == "para one\n\npara two"


def test_single_blank_line_between_paragraphs_is_kept():
    assert clean_text("para one\n\npara two") == "para one\n\npara two"


def test_leading_indentation_inside_text_is_kept():
    assert clean_text("code:\n    x = 1") == "code:\n    x = 1"


def test_whitespace_only_text_becomes_empty():
    assert clean_text("  \n\n \t ") == ""


def test_words_are_never_changed():
    text = "Hyphen-ated words, UPPER case & symbols: 42%"
    assert clean_text(text) == text


# --- ligatures --------------------------------------------------------------


def test_all_five_ligatures_become_plain_letters():
    assert replace_ligatures("ﬀ ﬁ ﬂ ﬃ ﬄ") == "ff fi fl ffi ffl"


def test_ligature_inside_word_is_fixed_by_clean_text():
    assert clean_text("eﬃcient") == "efficient"


def test_other_special_characters_are_not_changed():
    # Only ligatures are replaced; superscripts, dashes and quotes stay.
    assert replace_ligatures("x² – “quoted”") == "x² – “quoted”"


# --- hyphenated line breaks -------------------------------------------------


def test_word_split_by_line_break_is_rejoined():
    assert clean_text("passage re-\ntrieval works") == "passage retrieval works"


def test_real_hyphen_is_kept_when_document_uses_hyphenated_form():
    text = "A dual-encoder model.\nThe dual-\nencoder is fast."

    assert clean_text(text) == "A dual-encoder model.\nThe dual-encoder is fast."


def test_hyphen_removed_when_document_also_uses_joined_form():
    # Both "pre-training" and "pretraining" appear: the joined form wins.
    text = "pre-training and pretraining.\nWe use pre-\ntraining."

    assert clean_text(text).endswith("We use pretraining.")


def test_capitalized_second_part_is_left_alone():
    # "Lucene-\nBM25" is a name, not a split word.
    assert clean_text("Lucene-\nBM25 system") == "Lucene-\nBM25 system"


def test_hyphen_followed_by_space_is_not_a_split_word():
    assert clean_text("first- and second-order") == "first- and second-order"


def test_trailing_space_after_hyphen_still_rejoins():
    # Trailing spaces are removed before rejoining.
    assert clean_text("em-  \nbeddings") == "embeddings"


def test_vocabulary_from_another_page_is_used():
    # The page itself never writes "dual-encoder", but the document does.
    vocabulary = build_vocabulary("Page 1 says dual-encoder.")

    assert rejoin_hyphenated_words("dual-\nencoder", vocabulary) == "dual-encoder"


def test_vocabulary_counts_words_after_fixing_ligatures():
    vocabulary = build_vocabulary("eﬃcient multi-hop")

    assert "efficient" in vocabulary.plain_words
    assert "multi-hop" in vocabulary.hyphenated_words
