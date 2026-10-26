"""Light, safe text cleaning applied to every document before chunking.

Cleaning is deliberately minimal. Aggressive cleaning (removing symbols,
lowercasing, broad Unicode normalization) can change meaning or destroy
content such as code and formulas, so a rule is only added when inspecting
real output shows a clear need. The two PDF-specific rules below
(ligatures and line-end hyphenation) were added after inspecting the arXiv
corpus in Milestone 1, Phase 1.4.
"""

import re
from dataclasses import dataclass

# Three or more newlines in a row (two or more blank lines) are collapsed to
# exactly two newlines, which keeps paragraph breaks but removes big gaps.
MULTIPLE_BLANK_LINES = re.compile(r"\n{3,}")

# PDFs often store letter pairs like "fi" as ONE special character called a
# ligature. Searching for "efficient" would not find "eﬃcient", so each
# ligature is replaced with its plain letters. Only these five are replaced;
# broad Unicode normalization (NFKC) is avoided because it also changes
# characters such as "²" into "2".
LIGATURES = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
}

# A word split across two lines with a hyphen, e.g. "ef-\nficient".
# Group 1 is the part before the hyphen; it must end in a lowercase letter.
# Group 2 is the part after the line break; it must start lowercase, so
# names like "Lucene-\nBM25" are left alone.
HYPHEN_LINE_BREAK = re.compile(r"([A-Za-z]*[a-z])-\n([a-z][A-Za-z]*)")

# Words made only of letters, e.g. "efficient".
PLAIN_WORD = re.compile(r"[A-Za-z]+")
# Words that contain hyphens on a single line, e.g. "dual-encoder".
HYPHENATED_WORD = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)+")


@dataclass
class Vocabulary:
    """Words seen in a document, used to decide how to rejoin split words.

    Attributes:
        plain_words: Lowercased words without hyphens, e.g. "efficient".
        hyphenated_words: Lowercased words with hyphens, e.g. "dual-encoder".
    """

    plain_words: set
    hyphenated_words: set


def replace_ligatures(text: str) -> str:
    """Replace ligature characters (like "ﬁ") with plain letters ("fi")."""
    for ligature, letters in LIGATURES.items():
        text = text.replace(ligature, letters)
    return text


def build_vocabulary(text: str) -> Vocabulary:
    """Collect the words that appear in a text, in lowercase.

    Ligatures are replaced first so "eﬃcient" is counted as "efficient".
    A hyphenated word split across a line break is not counted, because the
    patterns never match across a newline.
    """
    text = replace_ligatures(text)

    plain_words = set()
    for word in PLAIN_WORD.findall(text):
        plain_words.add(word.lower())

    hyphenated_words = set()
    for word in HYPHENATED_WORD.findall(text):
        hyphenated_words.add(word.lower())

    return Vocabulary(plain_words=plain_words, hyphenated_words=hyphenated_words)


def choose_rejoined_word(first_part: str, second_part: str, vocabulary: Vocabulary) -> str:
    """Decide how to rejoin a word that was split with a hyphen at a line end.

    If the document elsewhere writes the word WITH a hyphen ("dual-encoder")
    and never without one ("dualencoder"), the hyphen is real and is kept.
    Otherwise the hyphen was only added for the line break and is removed
    ("ef" + "ficient" -> "efficient").
    """
    joined = first_part + second_part
    hyphenated = first_part + "-" + second_part
    hyphen_is_real = (
        hyphenated.lower() in vocabulary.hyphenated_words
        and joined.lower() not in vocabulary.plain_words
    )
    if hyphen_is_real:
        return hyphenated
    return joined


def rejoin_hyphenated_words(text: str, vocabulary: Vocabulary) -> str:
    """Rejoin every word that was split across a line with a hyphen.

    The text is rebuilt piece by piece: unchanged text between matches is
    copied as it is, and each match is replaced by the rejoined word.
    """
    pieces = []
    previous_end = 0
    for match in HYPHEN_LINE_BREAK.finditer(text):
        pieces.append(text[previous_end:match.start()])
        pieces.append(choose_rejoined_word(match.group(1), match.group(2), vocabulary))
        previous_end = match.end()
    pieces.append(text[previous_end:])
    return "".join(pieces)


def clean_text(text: str, vocabulary: Vocabulary | None = None) -> str:
    """Normalize whitespace and fix common PDF text problems.

    Steps:
        1. Convert Windows (\\r\\n) and old Mac (\\r) line endings to \\n.
        2. Remove null characters, which some PDFs contain.
        3. Replace ligature characters with plain letters.
        4. Remove spaces and tabs at the end of each line.
        5. Rejoin words split across lines with a hyphen.
        6. Collapse runs of blank lines into a single blank line.
        7. Remove whitespace at the very start and end of the text.

    Leading spaces on a line are kept, because they matter for indented
    content such as code blocks in Markdown.

    Args:
        text: Raw text from a loader.
        vocabulary: Words used to decide how to rejoin hyphenated words.
            PDF loading passes one vocabulary built from the WHOLE document,
            because each page is cleaned separately. If None, it is built
            from this text.

    Returns:
        The cleaned text. May be an empty string if the input had no content.
    """
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = text.replace("\x00", "")
    text = replace_ligatures(text)

    cleaned_lines = []
    for line in text.split("\n"):
        cleaned_lines.append(line.rstrip())
    text = "\n".join(cleaned_lines)

    if vocabulary is None:
        vocabulary = build_vocabulary(text)
    text = rejoin_hyphenated_words(text, vocabulary)

    text = MULTIPLE_BLANK_LINES.sub("\n\n", text)
    return text.strip()
