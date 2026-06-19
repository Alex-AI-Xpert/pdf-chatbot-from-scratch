"""
============================================================
chunker.py
============================================================

PURPOSE
-------
This module is responsible for STEP 2 of our pipeline:
    "Split the extracted PDF text into small, manageable
     pieces ('chunks'), while remembering metadata about
     each chunk (which page it came from, its position, etc.)"

WHY DO WE NEED CHUNKING?
-------------------------
Imagine your PDF is a 50-page textbook. If a user asks
"What is supervised learning?", we don't want to compare their
question against the ENTIRE 50-page document at once -- that's
too coarse. Instead, we break the document into small chunks
(e.g. ~200 words each) so that:

  1. Each chunk is focused on a smaller, more specific topic.
  2. Our similarity search (Step 4) can pinpoint exactly WHICH
     chunk(s) are most relevant to the question.
  3. We can tell the user precisely which page(s) the answer
     came from.

BEGINNER NOTE
-------------
Think of chunking like cutting a long essay into paragraphs
(or smaller groups of paragraphs) so that you can quickly
flip to the "right paragraph" instead of reading the whole essay.

CONFIGURABILITY
----------------
Two parameters control chunking:
  - chunk_size: how many WORDS go into each chunk
  - overlap: how many WORDS from the end of one chunk are
    repeated at the start of the next chunk.

Overlap is important because it prevents important sentences
from being awkwardly cut in half between two chunks, losing
context.
"""

from typing import List, Dict


def chunk_pages(
    pages_data: List[Dict],
    chunk_size: int = 200,
    overlap: int = 50,
) -> List[Dict]:
    """
    Split page-level text into smaller overlapping chunks.

    Parameters
    ----------
    pages_data : List[Dict]
        The output of `pdf_extractor.extract_text_from_pdf`.
        Each item looks like: {"page_number": int, "text": str}

    chunk_size : int, optional (default=200)
        The target number of WORDS per chunk. Smaller chunks
        give more precise retrieval but may lose context.
        Larger chunks give more context but less precision.

    overlap : int, optional (default=50)
        The number of WORDS that will be repeated between
        consecutive chunks (sliding window). This helps avoid
        cutting a sentence/idea in half between two chunks.

        NOTE: overlap must be smaller than chunk_size, otherwise
        the chunking would never make progress through the text.

    Returns
    -------
    List[Dict]
        A list of chunk dictionaries, each with the structure:

            {
                "chunk_id": int,          # unique ID, starting at 0
                "text": str,              # the chunk's text
                "page_number": int,       # which page this text mostly came from
                "word_count": int,        # number of words in this chunk
            }

    Example
    -------
    >>> pages = [{"page_number": 1, "text": "word " * 250}]
    >>> chunks = chunk_pages(pages, chunk_size=100, overlap=20)
    >>> len(chunks) > 1
    True
    """

    # Safety check: overlap must be smaller than chunk_size,
    # otherwise our sliding window would get stuck (never advance).
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})"
        )

    all_chunks = []
    chunk_id = 0

    # We process each page separately. This keeps the page_number
    # metadata accurate -- a chunk will never accidentally span
    # across two pages and "lose" the correct page reference.
    for page in pages_data:
        page_number = page["page_number"]
        words = page["text"].split()  # split text into a list of words

        # If a page has very little text (e.g. an almost-blank page),
        # we still create one small chunk for it, as long as it's
        # not completely empty.
        if len(words) == 0:
            continue

        start = 0
        while start < len(words):
            end = start + chunk_size

            # Slice out the words for this chunk
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            all_chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "page_number": page_number,
                "word_count": len(chunk_words),
            })

            chunk_id += 1

            # Move the "window" forward by (chunk_size - overlap) words.
            # This is what creates the overlapping effect:
            # e.g. if chunk_size=200 and overlap=50, we move forward
            # by 150 words each time, so the last 50 words of this
            # chunk will also appear at the start of the next chunk.
            start += (chunk_size - overlap)

    return all_chunks


def print_chunk_summary(chunks: List[Dict]) -> None:
    """
    Print a human-readable summary of the chunking results.

    This is a small debugging/teaching helper -- it's useful
    when running this module standalone in a notebook to
    "see" what chunking produced.

    Parameters
    ----------
    chunks : List[Dict]
        The output of `chunk_pages`.

    Returns
    -------
    None
        This function only prints information; it returns nothing.
    """
    print(f"Total chunks created: {len(chunks)}")
    for chunk in chunks[:3]:  # show only the first 3 as a preview
        preview = chunk["text"][:80]  # first 80 characters
        print(
            f"  Chunk {chunk['chunk_id']} "
            f"(page {chunk['page_number']}, "
            f"{chunk['word_count']} words): "
            f"\"{preview}...\""
        )
