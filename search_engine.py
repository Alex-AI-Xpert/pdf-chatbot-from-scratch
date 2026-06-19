"""
============================================================
search_engine.py
============================================================

PURPOSE
-------
This module handles STEPS 4 and 5 of our pipeline:
    Step 4: Convert the user's question into a TF-IDF vector,
            compute cosine similarity against every chunk,
            and retrieve the top matching chunk(s).
    Step 5: Return the most relevant chunk(s), highlight the
            matching words, and show similarity scores.

============================================================
THE MATH BEHIND COSINE SIMILARITY
============================================================
Recall from vectorizer.py that every chunk is represented as a
vector of numbers (its TF-IDF vector), and every vector has been
"L2-normalized" to have a length of 1.

Cosine similarity measures the ANGLE between two vectors, not
their magnitude (length). The formula is:

    cosine_similarity(A, B) = (A . B) / (||A|| * ||B||)

where:
    A . B   = dot product of A and B
              = sum( A_i * B_i  for all i )
    ||A||   = the L2 norm (length) of vector A
    ||B||   = the L2 norm (length) of vector B

BEGINNER INTUITION
-------------------
Imagine each chunk and each question as an arrow pointing in
some direction in a very high-dimensional space (one dimension
per unique word in the vocabulary).

- If two arrows point in almost the SAME direction, the angle
  between them is small, and cosine similarity is close to 1
  (very similar).
- If two arrows point in COMPLETELY DIFFERENT directions
  (90 degrees apart), cosine similarity is 0 (unrelated).
- If two arrows point in OPPOSITE directions, cosine similarity
  is -1 (but with TF-IDF, since all values are non-negative,
  similarity is always between 0 and 1).

SIMPLIFICATION FOR NORMALIZED VECTORS
---------------------------------------
Because our vectorizer already L2-normalizes every vector
(||A|| = 1 and ||B|| = 1), the formula simplifies to JUST the
dot product:

    cosine_similarity(A, B) = A . B

This is why, in the code below, we can compute similarity using
a simple matrix multiplication!
"""

from typing import List, Dict
import torch

from vectorizer import TfidfVectorizer, tokenize


def compute_similarities(
    question_vector: torch.Tensor,
    chunk_vectors: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the cosine similarity between a question vector and
    every chunk vector.

    Parameters
    ----------
    question_vector : torch.Tensor
        Shape (1, vocab_size) -- the TF-IDF vector for the
        user's question (already L2-normalized).

    chunk_vectors : torch.Tensor
        Shape (num_chunks, vocab_size) -- the TF-IDF vectors for
        every chunk (already L2-normalized).

    Returns
    -------
    torch.Tensor
        Shape (num_chunks,) -- the cosine similarity score
        between the question and each chunk, ranging from
        0.0 (completely unrelated) to 1.0 (identical direction).

    HOW THIS WORKS
    --------------
    Since both `question_vector` and `chunk_vectors` are already
    L2-normalized (length = 1), cosine similarity reduces to a
    simple dot product:

        similarity[i] = question_vector . chunk_vectors[i]

    We compute ALL of these dot products at once using matrix
    multiplication (torch.matmul), which is much faster than
    looping over each chunk individually.
    """
    # chunk_vectors:    shape (num_chunks, vocab_size)
    # question_vector:  shape (1, vocab_size)
    #
    # To multiply them, we transpose question_vector to shape
    # (vocab_size, 1), so that:
    #   (num_chunks, vocab_size) @ (vocab_size, 1) -> (num_chunks, 1)
    similarities = torch.matmul(chunk_vectors, question_vector.T)

    # Flatten from shape (num_chunks, 1) to shape (num_chunks,)
    return similarities.squeeze(dim=1)


def search(
    question: str,
    vectorizer: TfidfVectorizer,
    chunk_vectors: torch.Tensor,
    chunks: List[Dict],
    top_k: int = 3,
) -> List[Dict]:
    """
    Find the chunks most relevant to a user's question.

    This function implements STEP 4 of our pipeline.

    Parameters
    ----------
    question : str
        The user's natural-language question, e.g.
        "What is supervised learning?"

    vectorizer : TfidfVectorizer
        The SAME vectorizer object that was fit on the document
        chunks (see vectorizer.build_chunk_vectors). We reuse its
        vocabulary and IDF values so the question is represented
        in the SAME vector space as the chunks.

    chunk_vectors : torch.Tensor
        Shape (num_chunks, vocab_size) -- TF-IDF vectors for all
        chunks (from vectorizer.build_chunk_vectors).

    chunks : List[Dict]
        The original chunk dictionaries (from chunker.chunk_pages),
        used to retrieve the chunk text and page numbers.

    top_k : int, optional (default=3)
        How many top-matching chunks to return.

    Returns
    -------
    List[Dict]
        A list of up to `top_k` result dictionaries, sorted from
        most relevant to least relevant, each with the structure:

            {
                "chunk_id": int,
                "page_number": int,
                "text": str,
                "similarity_score": float,   # between 0.0 and 1.0
            }

    Example
    -------
    >>> results = search("What is AI?", vectorizer, chunk_vectors, chunks, top_k=2)
    >>> len(results) <= 2
    True
    >>> results[0]["similarity_score"] >= results[-1]["similarity_score"]
    True
    """
    # --- Step 1: Convert the question into a TF-IDF vector ---
    # We use `transform` (NOT `fit`), because we want to use the
    # SAME vocabulary and IDF values that were learned from the
    # document chunks. If we called `fit` again here, the question
    # would live in a DIFFERENT vector space and comparisons would
    # be meaningless.
    question_vector = vectorizer.transform([question])  # shape (1, vocab_size)

    # --- Step 2: Compute cosine similarity against every chunk ---
    similarities = compute_similarities(question_vector, chunk_vectors)

    # --- Step 3: Find the indices of the top_k highest scores ---
    # torch.topk returns both the values and their indices, sorted
    # from highest to lowest.
    k = min(top_k, len(chunks))  # don't ask for more chunks than exist
    top_scores, top_indices = torch.topk(similarities, k=k)

    # --- Step 4: Build the result list ---
    results = []
    for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
        chunk = chunks[idx]
        results.append({
            "chunk_id": chunk["chunk_id"],
            "page_number": chunk["page_number"],
            "text": chunk["text"],
            "similarity_score": score,
        })

    return results


def highlight_matching_words(chunk_text: str, question: str) -> str:
    """
    Wrap words from the question that also appear in the chunk
    with **double asterisks** (Markdown bold syntax), so the UI
    can visually highlight them.

    This implements the "highlight matching text" requirement
    from STEP 5.

    Parameters
    ----------
    chunk_text : str
        The raw text of a retrieved chunk.

    question : str
        The user's original question.

    Returns
    -------
    str
        The chunk text, with any word that also appears in the
        question wrapped in **double asterisks** for Markdown
        bold rendering (e.g. in Streamlit's `st.markdown`).

    Example
    -------
    >>> highlight_matching_words("Cats and dogs are pets", "What is a dog?")
    'Cats and **dogs** are pets'

    BEGINNER NOTE
    -------------
    This is a simple, word-by-word approach: we tokenize the
    question, build a set of its words, and then go through the
    chunk text word-by-word, wrapping any word that matches.
    We use `tokenize()` from vectorizer.py so that matching is
    done in a consistent, case-insensitive way.
    """
    question_words = set(tokenize(question))

    # We split on whitespace (not tokenize) here, because we want
    # to PRESERVE the original punctuation/casing of the chunk text
    # for display purposes -- we only use `tokenize` to DECIDE
    # whether a word matches.
    original_words = chunk_text.split()

    highlighted_words = []
    for word in original_words:
        # Strip punctuation from this word just for comparison
        # purposes (e.g. "dog," -> "dog")
        cleaned = tokenize(word)
        is_match = len(cleaned) > 0 and cleaned[0] in question_words

        if is_match:
            highlighted_words.append(f"**{word}**")
        else:
            highlighted_words.append(word)

    return " ".join(highlighted_words)
