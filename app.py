"""
============================================================
app.py
============================================================

PURPOSE
-------
This is the STREAMLIT USER INTERFACE for "PDF Chatbot From
Scratch" (Version 1). It ties together all the other modules:

    pdf_extractor.py  -> Step 1: Extract text from PDF
    chunker.py        -> Step 2: Split text into chunks
    vectorizer.py     -> Step 3: TF-IDF vectorization
    search_engine.py  -> Steps 4-5: Similarity search + answer display

HOW TO RUN THIS APP
--------------------
From a terminal, inside the project folder, run:

    streamlit run app.py

If you are running this in Google Colab, see the README.md for
instructions on using `streamlit` + a tunneling tool (like
`localtunnel` or `ngrok`) to view the app in your browser.

WHAT THIS APP DOES (USER'S PERSPECTIVE)
------------------------------------------
1. The user uploads a PDF file.
2. The app extracts text and splits it into chunks (Steps 1-2).
3. The app converts all chunks into TF-IDF vectors (Step 3).
4. The user types a question.
5. The app finds the most similar chunk(s) to the question
   (Step 4) and displays them with similarity scores, page
   numbers, and highlighted matching words (Step 5).
"""

import streamlit as st

# Import our custom modules. Because all files are in the same
# folder, Python can import them directly by filename (without
# the ".py" extension).
from pdf_extractor import extract_text_from_pdf, get_total_character_count
from chunker import chunk_pages
from vectorizer import build_chunk_vectors
from search_engine import search, highlight_matching_words


# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------
st.set_page_config(
    page_title="PDF Chatbot From Scratch",
    page_icon="📄",
    layout="wide",
)

st.title("📄 PDF Chatbot From Scratch")
st.caption(
    "An educational project demonstrating how a simple "
    "retrieval-based chatbot works: PDF extraction → chunking → "
    "TF-IDF vectors → cosine similarity search. No external LLM "
    "APIs are used!"
)


# ------------------------------------------------------------
# SIDEBAR: CONFIGURATION OPTIONS
# ------------------------------------------------------------
# These let students EXPERIMENT with how chunking and retrieval
# behave under different settings.
st.sidebar.header("⚙️ Settings")

chunk_size = st.sidebar.slider(
    "Chunk size (words per chunk)",
    min_value=50,
    max_value=500,
    value=200,
    step=10,
    help=(
        "How many words go into each chunk. Smaller chunks are "
        "more precise but may lose context. Larger chunks keep "
        "more context but are less precise."
    ),
)

overlap = st.sidebar.slider(
    "Overlap (words shared between chunks)",
    min_value=0,
    max_value=chunk_size - 10,
    value=min(50, chunk_size - 10),
    step=10,
    help=(
        "How many words are repeated between consecutive chunks. "
        "Helps avoid cutting sentences/ideas in half."
    ),
)

top_k = st.sidebar.slider(
    "Number of chunks to retrieve",
    min_value=1,
    max_value=10,
    value=3,
    help="How many of the most relevant chunks to show as the answer.",
)


# ------------------------------------------------------------
# STEP 1 & 2: PDF UPLOAD, EXTRACTION, AND CHUNKING
# ------------------------------------------------------------
st.header("1️⃣ Upload a PDF")

uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

# We use Streamlit's "session state" to remember the processed
# chunks and vectors across user interactions (e.g. when the
# user types a question, we don't want to re-process the PDF).
if "chunks" not in st.session_state:
    st.session_state.chunks = None
    st.session_state.vectorizer = None
    st.session_state.chunk_vectors = None
    st.session_state.last_settings = None


if uploaded_file is not None:
    # Track the current settings so we know if we need to
    # re-process the PDF (e.g. if the user changed chunk_size).
    current_settings = (uploaded_file.name, chunk_size, overlap)

    if st.session_state.last_settings != current_settings:
        with st.spinner("Extracting text from PDF..."):
            # --- STEP 1: Extract text from the PDF ---
            pages_data = extract_text_from_pdf(uploaded_file)
            total_chars = get_total_character_count(pages_data)

        with st.spinner("Splitting text into chunks..."):
            # --- STEP 2: Split text into chunks ---
            chunks = chunk_pages(pages_data, chunk_size=chunk_size, overlap=overlap)

        with st.spinner("Building TF-IDF vectors..."):
            # --- STEP 3: Convert chunks into TF-IDF vectors ---
            vectorizer, chunk_vectors = build_chunk_vectors(chunks)

        # Save everything in session state so we don't redo this
        # work every time the user asks a new question.
        st.session_state.chunks = chunks
        st.session_state.vectorizer = vectorizer
        st.session_state.chunk_vectors = chunk_vectors
        st.session_state.last_settings = current_settings

        st.success(
            f"✅ Processed '{uploaded_file.name}': "
            f"{len(pages_data)} pages, {total_chars:,} characters, "
            f"{len(chunks)} chunks created."
        )
    else:
        st.info(
            f"Using previously processed data for '{uploaded_file.name}' "
            f"({len(st.session_state.chunks)} chunks)."
        )


# ------------------------------------------------------------
# STEP 4 & 5: ASK A QUESTION AND DISPLAY RESULTS
# ------------------------------------------------------------
st.header("2️⃣ Ask a Question")

if st.session_state.chunks is None:
    st.warning("👆 Please upload a PDF first.")
else:
    question = st.text_input(
        "Type your question about the PDF:",
        placeholder="e.g. What is the main topic of this document?",
    )

    if question:
        with st.spinner("Searching for relevant chunks..."):
            # --- STEPS 4 & 5: Search + retrieve top chunks ---
            results = search(
                question=question,
                vectorizer=st.session_state.vectorizer,
                chunk_vectors=st.session_state.chunk_vectors,
                chunks=st.session_state.chunks,
                top_k=top_k,
            )

        st.subheader("📌 Most Relevant Chunks")

        if all(r["similarity_score"] == 0 for r in results):
            st.warning(
                "No chunks share any words with your question. "
                "Try rephrasing, or use different keywords from "
                "the document."
            )

        for rank, result in enumerate(results, start=1):
            score = result["similarity_score"]
            page = result["page_number"]

            # Highlight matching words for display (Step 5)
            highlighted_text = highlight_matching_words(result["text"], question)

            with st.container(border=True):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**Result #{rank} — Page {page}**")
                with col2:
                    st.markdown(f"**Similarity score:** `{score:.4f}`")

                # Show a progress bar as a visual representation of
                # the similarity score (0.0 to 1.0).
                st.progress(min(max(score, 0.0), 1.0))

                st.markdown(highlighted_text)


# ------------------------------------------------------------
# FOOTER: EXPLANATION FOR STUDENTS
# ------------------------------------------------------------
with st.expander("ℹ️ How does this chatbot work? (Click to expand)"):
    st.markdown(
        """
This chatbot does **NOT** use any large language model (LLM) to
"generate" answers. Instead, it works like a smart search engine
over your PDF:

1. **PDF Extraction**: We read the text out of your PDF, page by page.
2. **Chunking**: We split the text into overlapping word windows
   so we can pinpoint specific sections.
3. **TF-IDF Vectorization**: We convert each chunk (and your
   question) into a vector of numbers that represents which
   words are important.
4. **Cosine Similarity Search**: We compare your question's
   vector to every chunk's vector using cosine similarity, and
   return the chunks that are most "aligned" with your question.
5. **Answer Display**: We show you the raw text of the most
   relevant chunk(s), highlight words that overlap with your
   question, and show you the page number and similarity score.

This is sometimes called a **"retrieval" system** — it retrieves
relevant information rather than generating new text. Later
versions of this project (see roadmap) add semantic embeddings,
vector databases, and transformer-based answer generation to
build a full Retrieval-Augmented Generation (RAG) system.
        """
    )
