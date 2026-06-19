"""
============================================================
vectorizer.py
============================================================

PURPOSE
-------
This module is responsible for STEP 3 of our pipeline:
    "Convert each text chunk into a numerical vector using
     TF-IDF (Term Frequency - Inverse Document Frequency),
     so that a computer can compare chunks mathematically."

WHY DO WE NEED THIS?
---------------------
Computers can't compare "meaning" directly -- they can only
compare NUMBERS. TF-IDF is a classic, simple, and surprisingly
effective way to turn text into numbers that capture how
"important" each word is to each chunk.

============================================================
THE MATH BEHIND TF-IDF (explained step by step)
============================================================

Imagine we have a "vocabulary" V containing every unique word
that appears across ALL chunks (this is sometimes called the
"corpus"). Every chunk will be represented as a vector with
one number per word in the vocabulary.

--------------------------------------------------------------
1) TERM FREQUENCY (TF)
--------------------------------------------------------------
   TF(word, chunk) = (number of times "word" appears in "chunk")
                      ---------------------------------------------
                      (total number of words in "chunk")

   Intuition: if the word "neural" appears 5 times in a 100-word
   chunk, its TF score is 5/100 = 0.05. TF tells us how FREQUENT
   a word is WITHIN a single chunk.

--------------------------------------------------------------
2) INVERSE DOCUMENT FREQUENCY (IDF)
--------------------------------------------------------------
   IDF(word) = log( (1 + N) / (1 + DF(word)) ) + 1

   where:
     N       = total number of chunks (documents) in our corpus
     DF(word) = number of chunks that contain "word" at least once

   Intuition: common words like "the", "is", "and" appear in
   almost every chunk, so DF(word) is close to N, and the ratio
   (1+N)/(1+DF) is close to 1, so log(...) is close to 0.
   That means common words get a LOW IDF score (they're not
   very "special" or informative).

   Rare, specific words (e.g. "backpropagation") appear in only
   a few chunks, so DF(word) is small, the ratio (1+N)/(1+DF) is
   large, and log(...) is large. That means rare words get a HIGH
   IDF score (they're very informative -- they help distinguish
   one chunk from another).

   (The "+1"s in the formula are called "smoothing" -- they
   prevent division by zero and prevent IDF from being exactly 0.)

--------------------------------------------------------------
3) TF-IDF SCORE
--------------------------------------------------------------
   TFIDF(word, chunk) = TF(word, chunk) * IDF(word)

   This combines BOTH ideas:
     - "How often does this word appear in THIS chunk?" (TF)
     - "How rare/special is this word ACROSS ALL chunks?" (IDF)

   A word gets a HIGH TF-IDF score in a chunk if it appears
   often in that chunk AND it's rare across the whole document.
   Such words are the "fingerprint" of that chunk's topic.

--------------------------------------------------------------
4) L2 NORMALIZATION
--------------------------------------------------------------
   After computing the raw TF-IDF vector for a chunk, we divide
   every value by the vector's "length" (its L2 norm):

       norm = sqrt( sum( value_i^2 for all i ) )
       normalized_vector_i = value_i / norm

   This makes every chunk's vector have a length of 1.0,
   regardless of how long the chunk's text is. This is important
   for cosine similarity (Step 4) to work fairly -- a very long
   chunk shouldn't automatically "win" just because it has bigger
   numbers.

============================================================
WHY USE PYTORCH HERE?
============================================================
We use PyTorch tensors (instead of plain Python lists or NumPy
arrays) to store our TF-IDF vectors because:
  1. It's good practice for students moving toward deep learning
     (Versions 2-5 of this project use PyTorch heavily for
     embeddings and transformers).
  2. PyTorch makes matrix operations (like computing TF-IDF for
     ALL chunks at once, or cosine similarity) fast and concise.
"""

from typing import List, Dict, Tuple
import math
import re
import torch


def tokenize(text: str) -> List[str]:
    """
    Convert a string of text into a list of lowercase word tokens.

    Parameters
    ----------
    text : str
        Raw text, e.g. "Neural Networks are AMAZING!"

    Returns
    -------
    List[str]
        A list of lowercase word tokens, with punctuation removed.
        e.g. ["neural", "networks", "are", "amazing"]

    BEGINNER NOTE
    -------------
    "Tokenization" just means "breaking text into smaller pieces
    (tokens)" -- here, our tokens are simply words.
    We lowercase everything so that "Neural" and "neural" are
    treated as the SAME word.
    """
    # \b\w+\b matches "word characters" (letters, digits, underscore)
    # surrounded by word boundaries. This effectively strips out
    # punctuation like commas, periods, quotes, etc.
    text = text.lower()
    tokens = re.findall(r"\b\w+\b", text)
    return tokens


class TfidfVectorizer:
    """
    A simple, from-scratch TF-IDF vectorizer built with PyTorch.

    This class is intentionally simple compared to libraries like
    scikit-learn's TfidfVectorizer -- the goal is for students to
    be able to read every line and understand exactly what's
    happening mathematically.

    Usage
    -----
    >>> vectorizer = TfidfVectorizer()
    >>> vectorizer.fit(["the cat sat", "the dog ran"])
    >>> vectors = vectorizer.transform(["the cat sat", "the dog ran"])
    >>> vectors.shape
    torch.Size([2, 4])
    """

    def __init__(self):
        # self.vocabulary maps each unique word -> a column index
        # in our TF-IDF vectors. E.g. {"cat": 0, "dog": 1, "ran": 2, "sat": 3}
        self.vocabulary: Dict[str, int] = {}

        # self.idf_vector stores the IDF score for each word in
        # the vocabulary (same order as self.vocabulary indices).
        self.idf_vector: torch.Tensor = torch.tensor([])

    def fit(self, documents: List[str]) -> None:
        """
        Learn the vocabulary and IDF scores from a list of documents.

        "Fitting" means: look at ALL the documents (chunks) once,
        build the vocabulary, and compute how rare/common each
        word is (IDF), so we can later convert any text into a
        TF-IDF vector using `transform()`.

        Parameters
        ----------
        documents : List[str]
            A list of text chunks (raw strings).

        Returns
        -------
        None
            This method updates `self.vocabulary` and
            `self.idf_vector` in place.
        """
        n_documents = len(documents)

        # --- Step A: Build the vocabulary ---
        # We tokenize every document and collect the SET of unique
        # words. Then we assign each unique word an index (0, 1, 2, ...)
        tokenized_docs = [tokenize(doc) for doc in documents]

        unique_words = set()
        for tokens in tokenized_docs:
            unique_words.update(tokens)

        # Sort for a deterministic (repeatable) ordering.
        sorted_words = sorted(unique_words)
        self.vocabulary = {word: idx for idx, word in enumerate(sorted_words)}
        vocab_size = len(self.vocabulary)

        # --- Step B: Compute Document Frequency (DF) for each word ---
        # DF(word) = number of documents that contain "word" at least once
        df_counts = torch.zeros(vocab_size)

        for tokens in tokenized_docs:
            # Use a set() so that if a word appears multiple times
            # in one document, it only counts ONCE toward DF.
            unique_tokens_in_doc = set(tokens)
            for word in unique_tokens_in_doc:
                df_counts[self.vocabulary[word]] += 1

        # --- Step C: Compute IDF for each word ---
        # IDF(word) = log( (1 + N) / (1 + DF(word)) ) + 1
        #
        # This is the "smoothed" IDF formula (the same one used by
        # scikit-learn by default). The "+1"s avoid division by zero
        # and ensure IDF is never negative or zero.
        n = torch.tensor(float(n_documents))
        self.idf_vector = torch.log((1 + n) / (1 + df_counts)) + 1.0

    def transform(self, documents: List[str]) -> torch.Tensor:
        """
        Convert a list of documents into TF-IDF vectors.

        This must be called AFTER `fit()`. `fit()` learns the
        vocabulary and IDF values; `transform()` applies them to
        (possibly new) documents.

        Parameters
        ----------
        documents : List[str]
            A list of text chunks (raw strings) to convert.

        Returns
        -------
        torch.Tensor
            A 2D tensor of shape (num_documents, vocab_size), where
            each row is the L2-normalized TF-IDF vector for one
            document.

            If a document contains ONLY words that are not in the
            vocabulary (e.g. it's empty, or the vectorizer was
            fit on different text), its vector will be all zeros.
        """
        vocab_size = len(self.vocabulary)
        n_documents = len(documents)

        # Create an empty matrix to hold the TF-IDF vectors.
        # Shape: (number of documents) x (vocabulary size)
        tfidf_matrix = torch.zeros((n_documents, vocab_size))

        for doc_idx, doc in enumerate(documents):
            tokens = tokenize(doc)
            total_words = len(tokens)

            if total_words == 0:
                # Empty document -> leave its vector as all zeros.
                continue

            # --- Step A: Compute raw term counts for this document ---
            word_counts: Dict[str, int] = {}
            for word in tokens:
                if word in self.vocabulary:  # ignore unknown words
                    word_counts[word] = word_counts.get(word, 0) + 1

            # --- Step B: Compute TF and TF-IDF for each word ---
            for word, count in word_counts.items():
                col = self.vocabulary[word]

                # TF(word, doc) = count of word in doc / total words in doc
                tf = count / total_words

                # IDF(word) was precomputed during fit()
                idf = self.idf_vector[col]

                # TF-IDF(word, doc) = TF * IDF
                tfidf_matrix[doc_idx, col] = tf * idf

            # --- Step C: L2-normalize this document's vector ---
            # This makes the vector's "length" equal to 1, so that
            # cosine similarity later only measures DIRECTION
            # (which words matter), not magnitude (how long the text is).
            row = tfidf_matrix[doc_idx]
            norm = torch.norm(row, p=2)  # L2 norm = sqrt(sum(value^2))

            if norm > 0:
                tfidf_matrix[doc_idx] = row / norm

        return tfidf_matrix

    def fit_transform(self, documents: List[str]) -> torch.Tensor:
        """
        Convenience method that calls `fit()` followed by `transform()`.

        Parameters
        ----------
        documents : List[str]
            A list of text chunks (raw strings).

        Returns
        -------
        torch.Tensor
            The TF-IDF matrix for these documents (see `transform`).
        """
        self.fit(documents)
        return self.transform(documents)


def build_chunk_vectors(chunks: List[Dict]) -> Tuple[TfidfVectorizer, torch.Tensor]:
    """
    Build TF-IDF vectors for a list of chunk dictionaries.

    This is a small "glue" function that connects the chunking
    step (Step 2) with the vectorization step (Step 3).

    Parameters
    ----------
    chunks : List[Dict]
        The output of `chunker.chunk_pages`. Each chunk dict must
        have a "text" key.

    Returns
    -------
    Tuple[TfidfVectorizer, torch.Tensor]
        - The fitted TfidfVectorizer (so we can transform new
          questions later using the SAME vocabulary/IDF values).
        - A tensor of shape (num_chunks, vocab_size) containing
          the TF-IDF vector for each chunk.

    Example
    -------
    >>> chunks = [{"text": "the cat sat"}, {"text": "the dog ran"}]
    >>> vectorizer, vectors = build_chunk_vectors(chunks)
    >>> vectors.shape[0] == len(chunks)
    True
    """
    texts = [chunk["text"] for chunk in chunks]
    vectorizer = TfidfVectorizer()
    chunk_vectors = vectorizer.fit_transform(texts)
    return vectorizer, chunk_vectors
