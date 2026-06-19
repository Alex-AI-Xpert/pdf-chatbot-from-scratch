# 📄 PDF Chatbot From Scratch — Version 1

An **educational** project that builds a PDF question-answering chatbot
**without using any proprietary LLM APIs** (no OpenAI, no Gemini, no
Claude/Anthropic, no Cohere, etc.).

Version 1 implements a classic **TF-IDF + cosine similarity retrieval
system** — the same core idea behind many real-world search engines —
built with **Python and PyTorch**, and wrapped in a **Streamlit** UI.

---

## 📁 Folder Structure

```
pdf_chatbot_from_scratch/
├── app.py              # Streamlit UI (entry point)
├── pdf_extractor.py     # Step 1: Extract text from PDF
├── chunker.py            # Step 2: Split text into chunks
├── vectorizer.py         # Step 3: TF-IDF vectorization (PyTorch)
├── search_engine.py      # Steps 4-5: Cosine similarity search + display
├── requirements.txt      # Python dependencies
└── README.md              # This file
```

---

## 🔧 Installation

### Option A: Local machine

```bash
# 1. Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate    # on Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`)
in your browser.

### Option B: Google Colab

Streamlit apps don't display natively inside a Colab cell, but you can
run the app and expose it through a tunnel. Here's a common approach
using `localtunnel`:

```python
# Cell 1: Install dependencies
!pip install -r requirements.txt
!npm install -g localtunnel
```

```python
# Cell 2: Upload all the .py files (pdf_extractor.py, chunker.py,
# vectorizer.py, search_engine.py, app.py) into the Colab working
# directory (e.g. via the Files panel, or using the upload widget).
```

```python
# Cell 3: Run Streamlit in the background, then open a tunnel
import subprocess
subprocess.Popen(["streamlit", "run", "app.py", "--server.port", "8501"])

!npx localtunnel --port 8501
```

`localtunnel` will print a public URL (e.g. `https://xxxx.loca.lt`) —
open that link to use the app. The first time, it may ask you to
click "Click to Continue" to verify the tunnel.

> 💡 **Tip:** If `localtunnel` doesn't work in your environment, `ngrok`
> is a popular alternative (`pip install pyngrok`).

---

## 🧠 Step-by-Step: How the System Works

### Step 1 — PDF Processing (`pdf_extractor.py`)
- We use `PyPDF2` to open the PDF and read text **page by page**.
- Each page's text is stored along with its **page number**, so we
  never lose track of *where* information came from.

### Step 2 — Text Chunking (`chunker.py`)
- Each page's text is split into **words**.
- We slide a "window" of `chunk_size` words across the text, moving
  forward by `chunk_size - overlap` words each time.
- This creates **overlapping chunks**, so ideas that span a chunk
  boundary aren't completely lost.
- Each chunk stores: `chunk_id`, `text`, `page_number`, `word_count`.

### Step 3 — TF-IDF Vectorization (`vectorizer.py`)
- We build a **vocabulary** of every unique word across all chunks.
- For each (word, chunk) pair, we compute:
  - **TF** (Term Frequency): how often the word appears *in this chunk*,
    relative to the chunk's total word count.
  - **IDF** (Inverse Document Frequency): how *rare* the word is
    *across all chunks* — common words get a low score, rare/specific
    words get a high score.
  - **TF-IDF = TF × IDF**: high when a word is frequent *here* but rare
    *everywhere else* — i.e., it's a good "fingerprint" for this chunk.
- Each chunk's vector is **L2-normalized** (scaled to length 1), which
  makes comparing chunks of different lengths fair.
- All of this is implemented with **PyTorch tensors**.

### Step 4 — Similarity Search (`search_engine.py`)
- The user's question is converted into a TF-IDF vector using the
  **same vocabulary and IDF values** learned from the document
  (this is critical — otherwise the question and chunks would live
  in different "spaces" and couldn't be compared).
- Because all vectors are L2-normalized, **cosine similarity reduces
  to a simple dot product**, computed for all chunks at once via
  `torch.matmul`.
- We use `torch.topk` to find the `top_k` chunks with the highest
  similarity scores.

### Step 5 — Answer Generation (`search_engine.py` + `app.py`)
- "Generation" here means **retrieval**, not text generation by a
  language model.
- We return the top matching chunk(s), each with:
  - The chunk's **text**
  - Its **page number**
  - Its **similarity score** (0.0 to 1.0)
  - **Highlighted words** that also appear in the user's question
    (rendered in **bold** via Markdown)

### UI (`app.py`)
- Streamlit handles file upload, configuration sliders (chunk size,
  overlap, number of results), the question input box, and displaying
  results with progress bars for similarity scores.
- `st.session_state` caches the processed PDF so re-asking questions
  doesn't re-run extraction/chunking/vectorization every time.

---

## 🗺️ Roadmap: Future Versions

This project is designed to be extended incrementally. Each version
introduces ONE new core AI concept, building toward a full modern RAG
(Retrieval-Augmented Generation) pipeline — **still without relying on
proprietary LLM APIs** (open-source / locally-run models can be used
in later versions).

### Version 2 — Semantic Embeddings
- Replace TF-IDF (which only matches exact words) with **dense
  embeddings** from a neural network (e.g. a small Transformer
  encoder trained/run locally with PyTorch, or an open-source
  sentence-embedding model).
- This allows the chatbot to match **meaning**, not just keywords —
  e.g. a question about "automobiles" could match a chunk about "cars".
- Concepts introduced: embedding layers, vector spaces, semantic
  similarity vs. lexical similarity.

### Version 3 — Vector Database
- Store chunk embeddings in a proper **vector database** (e.g. FAISS,
  Chroma, or a simple custom index built with PyTorch tensors).
- Introduces **approximate nearest neighbor (ANN) search** for fast
  retrieval over large documents/collections.
- Concepts introduced: indexing, ANN algorithms (e.g. IVF, HNSW),
  scaling retrieval to large corpora.

### Version 4 — Retrieval-Augmented Generation (RAG)
- Combine retrieval (Versions 1-3) with a **generative model**: take
  the retrieved chunks, build a prompt that includes them as context,
  and have a model generate a natural-language answer **grounded in
  the retrieved text**.
- Concepts introduced: prompt construction, context windows,
  "grounding" generated answers in retrieved evidence, hallucination
  reduction.

### Version 5 — Transformer-Based Answer Generation
- Replace the simple "show the matching chunk" approach with a real
  **Transformer decoder** (built or fine-tuned with PyTorch, e.g. a
  small open-source model like GPT-2 or a distilled model) that reads
  the retrieved chunks and **generates a fluent, summarized answer**.
- Concepts introduced: self-attention, encoder/decoder architectures,
  tokenization for generative models, fine-tuning, decoding strategies
  (greedy, beam search, sampling).

---

## 📝 Notes for Students

- Every function in this project has a **docstring** explaining its
  purpose, parameters, and return values.
- The code prioritizes **clarity over performance** — there are
  faster/more compact ways to implement TF-IDF (e.g.
  `sklearn.feature_extraction.text.TfidfVectorizer`), but writing it
  from scratch helps you understand exactly what's happening
  mathematically.
- Try changing `chunk_size`, `overlap`, and `top_k` in the sidebar and
  observe how the results change — this is a great way to build
  intuition for retrieval systems!
