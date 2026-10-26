"""Test helper that writes tiny, real PDF files.

Generating PDFs inside the tests means no binary fixture files need to be
committed, and each test can choose exactly what text each page contains.
"""

from pathlib import Path


def escape_pdf_text(text: str) -> str:
    """Escape characters that have special meaning inside a PDF text string."""
    text = text.replace("\\", "\\\\")
    text = text.replace("(", "\\(")
    text = text.replace(")", "\\)")
    return text


def write_simple_pdf(path: Path, page_texts: list) -> None:
    """Write a PDF with one line of text per page.

    An empty string makes a page with no text at all, like a scanned page.

    Args:
        path: Where to write the PDF.
        page_texts: The text for each page, in order.
    """
    # PDF objects, numbered from 1. Objects 1-3 are fixed; each page then
    # adds two objects: the page itself and its content stream.
    objects = []
    page_count = len(page_texts)

    page_object_numbers = []
    for index in range(page_count):
        page_object_numbers.append(4 + index * 2)
    kids = ""
    for number in page_object_numbers:
        kids = kids + f"{number} 0 R "

    objects.append("<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {page_count} >>")
    objects.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    for index in range(page_count):
        content_number = page_object_numbers[index] + 1
        objects.append(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_number} 0 R >>"
        )
        text = page_texts[index]
        if text == "":
            stream = ""
        else:
            stream = f"BT /F1 12 Tf 72 720 Td ({escape_pdf_text(text)}) Tj ET"
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")

    # Write the objects, remembering the byte position of each one for the
    # cross-reference table that PDF readers use to find objects.
    output = "%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(output.encode("latin-1")))
        output = output + f"{number} 0 obj\n{body}\nendobj\n"

    xref_position = len(output.encode("latin-1"))
    output = output + f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n"
    for offset in offsets:
        output = output + f"{offset:010d} 00000 n \n"
    output = output + (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_position}\n%%EOF\n"
    )

    path.write_bytes(output.encode("latin-1"))
