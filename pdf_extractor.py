"""
============================================================
pdf_extractor.py
============================================================

PURPOSE
-------
This module is responsible for STEP 1 of our pipeline:
    "Read a PDF file and turn it into plain text,
     while remembering which page each piece of text
     came from."

WHY DO WE NEED THIS?
---------------------
A PDF is a binary file format -- you can't just open it like a
.txt file and read words out of it. We need a library
(PyPDF2) that knows how to "decode" the PDF structure and
pull out the text stream for each page.

We keep track of page numbers because later, when we answer a
user's question, we want to be able to say:
    "This answer came from page 4 of your PDF."

BEGINNER NOTE
-------------
Think of a PDF like a sealed box of printed pages. PyPDF2 is the
tool that opens the box, page by page, and reads the text on
each page out loud to us.
"""

from typing import List, Dict
import PyPDF2


def extract_text_from_pdf(pdf_file) -> List[Dict]:
    """
    Extract text from a PDF file, page by page.

    Parameters
    ----------
    pdf_file : file-like object
        This can be:
          - a file path (string), OR
          - a file-like object (e.g. the object returned by
            Streamlit's `st.file_uploader`, or Python's `open()`)

    Returns
    -------
    List[Dict]
        A list of dictionaries, one per page, in the form:

            [
                {"page_number": 1, "text": "...text of page 1..."},
                {"page_number": 2, "text": "...text of page 2..."},
                ...
            ]

        We use a list of dictionaries (instead of one giant string)
        so that we NEVER lose the information about which page a
        piece of text came from. This page information will be
        carried forward into the chunking step.

    Example
    -------
    >>> pages = extract_text_from_pdf("my_notes.pdf")
    >>> pages[0]["page_number"]
    1
    >>> isinstance(pages[0]["text"], str)
    True
    """

    # PyPDF2.PdfReader can accept either a path string or a
    # file-like object (BytesIO), which makes this function
    # flexible enough to work both:
    #   - in plain Python scripts (pdf_file = "file.pdf")
    #   - in Streamlit (pdf_file = uploaded file object)
    reader = PyPDF2.PdfReader(pdf_file)

    pages_data = []

    # enumerate(..., start=1) gives us page numbers starting at 1
    # (humans count pages starting at 1, not 0!)
    for page_number, page in enumerate(reader.pages, start=1):

        # extract_text() pulls the raw text content out of the page.
        # If a page has no extractable text (e.g. it's a scanned
        # image), extract_text() may return None or an empty string.
        raw_text = page.extract_text() or ""

        # We do a very light cleanup here: collapse multiple
        # whitespace characters into single spaces, and strip
        # leading/trailing whitespace. This makes later text
        # processing (chunking, TF-IDF) more predictable.
        cleaned_text = " ".join(raw_text.split())

        pages_data.append({
            "page_number": page_number,
            "text": cleaned_text,
        })

    return pages_data


def get_full_text(pages_data: List[Dict]) -> str:
    """
    Combine all pages' text into one big string.

    This is mostly useful for debugging or for showing the user
    a quick summary of how much text was extracted.

    Parameters
    ----------
    pages_data : List[Dict]
        The output of `extract_text_from_pdf`.

    Returns
    -------
    str
        All page texts concatenated together, separated by spaces.

    Example
    -------
    >>> pages = [{"page_number": 1, "text": "Hello"},
    ...           {"page_number": 2, "text": "World"}]
    >>> get_full_text(pages)
    'Hello World'
    """
    return " ".join(page["text"] for page in pages_data)


def get_total_character_count(pages_data: List[Dict]) -> int:
    """
    Count the total number of characters extracted from the PDF.

    This is a small utility function, useful for displaying
    "We extracted X characters from your PDF" in the UI.

    Parameters
    ----------
    pages_data : List[Dict]
        The output of `extract_text_from_pdf`.

    Returns
    -------
    int
        Total number of characters across all pages.
    """
    return sum(len(page["text"]) for page in pages_data)
