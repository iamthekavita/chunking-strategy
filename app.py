## Add Timer - 22 June##
# 23 June #
import io
import hashlib
import math
import time
import requests
from typing import List

import streamlit as st
from PyPDF2 import PdfReader
from sklearn.metrics.pairwise import cosine_similarity

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Why LLMs Fail on Long Documents",
    layout="wide"
)

st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --card: #ffffff;
    --ink: #1f2a37;
    --muted: #59667a;
    --accent: #e4572e;
    --line: #e7eaf0;
}

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

.stApp {
    background: #ffffff;
    color: var(--ink);
}

.block-container {
    padding-top: 1.4rem;
    padding-bottom: 1.8rem;
}

h1, h2, h3 {
    letter-spacing: -0.02em;
}

.hero {
    max-width: 760px;
    margin: 0 auto 1.75rem auto;
    text-align: center;
}

.hero h1 {
    margin-bottom: 0.85rem;
}

.hero p {
    color: var(--muted);
    font-size: 1.02rem;
    line-height: 1.7;
    margin: 0;
}

.hero ul {
    display: inline-block;
    text-align: left;
    margin: 1rem 0 0 0;
    padding-left: 1.25rem;
    color: var(--ink);
}

.hero-caption {
    display: block;
    margin-top: 0.9rem;
    color: var(--muted);
    font-size: 0.92rem;
}

[data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 1rem 1rem 0.7rem 1rem;
    box-shadow: 0 4px 16px rgba(29, 45, 62, 0.06);
}

[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 0.35rem 0.5rem;
}

div.stButton > button {
    background: linear-gradient(135deg, #e4572e, #ff7a45);
    color: white;
    border: 0;
    border-radius: 10px;
    font-weight: 700;
    letter-spacing: 0.01em;
    box-shadow: 0 8px 20px rgba(228, 87, 46, 0.28);
}

div.stButton > button:hover {
    transform: translateY(-1px);
    transition: 0.2s ease;
}

.stTextArea textarea, .stTextInput input {
    border-radius: 10px;
}

@media (max-width: 900px) {
    .block-container {
        padding-top: 1rem;
        padding-left: 0.8rem;
        padding-right: 0.8rem;
    }
}
</style>
        """,
        unsafe_allow_html=True,
)

# ==================================================
# HEADER
# ==================================================

st.markdown(
    """
<div class="hero">
    <h1>Why LLMs Fail on Long or Complex Tasks</h1>
    <p>This demo shows how chunking helps Large Language Models (LLMs) handle large documents more effectively.</p>
    <ul>
        <li>❌ No Chunking</li>
        <li>✅ Chunking + Retrieval</li>
    </ul>
    <span class="hero-caption">Interactive demo for showcasing context limits, retrieval, and chunk quality.</span>
</div>
    """,
    unsafe_allow_html=True,
)

# ==================================================
# LAYOUT
# ==================================================

left_col, right_col = st.columns([1, 2])

# ==================================================
# LEFT PANEL
# ==================================================

with left_col:
    st.header("⚙️ Configuration")
    st.caption("Set up the document input and retrieval pipeline before running the demo.")

    st.markdown("#### 1) Upload Document")

    uploaded_pdf = st.file_uploader(
        "Upload PDF / Book",
        type=["pdf"]
    )

    # reset cached state when a new file is uploaded (UI-only workflow)
    if uploaded_pdf is not None:
        uploaded_name = getattr(uploaded_pdf, "name", None)
        if st.session_state.get("uploaded_name") != uploaded_name:
            st.session_state.uploaded_name = uploaded_name
            st.session_state.document_text = None
            st.session_state.chunks = []
            st.session_state.embedding_model = None
            st.session_state.chunk_embeddings = None
            st.session_state.retriever_key = None
            st.session_state.last_timing = {}

    st.divider()

    # Hardcoded defaults for cleaner UI
    embedding_model = "nomic-embed-text"
    llama_model = "llama3.1"
    ollama_url = "http://localhost:11434"

    # Top checkbox as requested
    st.markdown("#### 2) Retrieval Settings")
    enable_overlap = st.checkbox("Chunking with Overlap", value=False)

    overlap_words = 0
    if enable_overlap:
        overlap_words = st.selectbox("Overlap Words", [20, 50, 75, 100, 125])

    st.divider()

    st.markdown("#### 3) Chunking Strategy")

    chunk_strategy = st.selectbox(
        "Chunking Strategy",
        [
            "Index (equal slices)",
            "Paragraph (natural paragraphs)",
            "Fixed-size (short chunks)",
            "Fixed-size (long chunks)"
        ],
        index=0,
    )

    num_chunks = st.selectbox("Top Chunks to Send to LLM", list(range(1, 11)), index=4)

    st.divider()

    st.markdown("#### 4) Quick Stats")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Top Chunks", num_chunks)

    with col2:
        st.metric("Overlap", overlap_words)


    # initialize session state
    if "document_text" not in st.session_state:
        st.session_state.document_text = None
    if "chunks" not in st.session_state:
        st.session_state.chunks = []
    if "embedding_model" not in st.session_state:
        st.session_state.embedding_model = None
    if "chunk_embeddings" not in st.session_state:
        st.session_state.chunk_embeddings = None
    if "retriever_key" not in st.session_state:
        st.session_state.retriever_key = None
    if "last_timing" not in st.session_state:
        st.session_state.last_timing = {}

# ==================================================
# RIGHT PANEL
# ==================================================

with right_col:

    st.header("💬 User Question")
    st.caption("Ask a question about the uploaded document. The app retrieves the top selected chunks and sends them to the LLM.")

    user_question = st.text_area(
        "",
        placeholder="Ask something about the document...",
        height=120,
        disabled=(uploaded_pdf is None)
    )

    chatbot_answer_slot = st.container()
    run_btn = st.button("🚀 Run Demo", use_container_width=True)
    st.caption("Click Run Demo to process the document using the selected settings.")


def extract_text_from_pdf(uploaded_file) -> str:
    try:
        # uploaded_file may be raw bytes or a file-like object
        if hasattr(uploaded_file, "read"):
            data = uploaded_file.read()
        else:
            data = uploaded_file
        reader = PdfReader(io.BytesIO(data))
        pages = []
        for p in reader.pages:
            text = p.extract_text() or ""
            pages.append(text)
        return "\n".join(pages)
    except Exception:
        return ""


def compute_stats(text: str, reader: PdfReader = None) -> dict:
    words = text.split()
    pages = None
    try:
        if reader is not None:
            pages = len(reader.pages)
    except Exception:
        pages = None
    words_count = len(words)
    tokens_est = int(words_count * 1.3)
    return {"pages": pages or "Unknown", "words": words_count, "tokens": tokens_est}


def create_chunks(text: str, overlap_words: int, strategy: str) -> List[str]:
    words = text.split()
    if len(words) == 0:
        return []

    # Paragraph strategy: keep natural paragraph boundaries.
    if strategy.startswith("Paragraph"):
        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paras:
            paras = [p.strip() for p in text.split("\n") if p.strip()]
        return paras

    # Fixed-size strategies decide chunk size, not chunk count.
    if strategy.startswith("Fixed-size"):
        if "short" in strategy.lower():
            preferred = 200
        else:
            preferred = 800
        chunks = []
        i = 0
        while i < len(words):
            chunk = " ".join(words[i : i + preferred])
            chunks.append(chunk)
            i += preferred - overlap_words if overlap_words > 0 else preferred
        return chunks

    # Index strategy uses evenly sized windows with a fixed target size.
    base = 400

    if overlap_words >= base:
        overlap_words = max(0, base - 1)

    chunks = []
    step = base - overlap_words if overlap_words > 0 else base
    if step <= 0:
        step = 1

    start = 0
    while start < len(words):
        end = start + base
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += step

    return chunks


def get_ollama_embedding(text: str, model: str, ollama_api_url: str) -> List[float]:
    base_url = ollama_api_url.rstrip("/")
    endpoint = f"{base_url}/api/embeddings"

    response = requests.post(
        endpoint,
        json={"model": model, "prompt": text},
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    embedding = payload.get("embedding")
    if not embedding:
        raise ValueError("No embedding returned by Ollama")
    return embedding


def build_retriever(chunks: List[str], embedding_model: str, ollama_api_url: str):
    if not chunks:
        return None, None

    chunk_embeddings = []
    total_chunks = len(chunks)
    progress = st.progress(0, text="Generating chunk embeddings...")
    for idx, chunk in enumerate(chunks, start=1):
        if chunk.strip():
            chunk_embeddings.append(get_ollama_embedding(chunk, embedding_model, ollama_api_url))
        else:
            chunk_embeddings.append([])
        progress.progress(int((idx / total_chunks) * 100), text=f"Generating chunk embeddings... {idx}/{total_chunks}")

    progress.empty()

    return embedding_model, chunk_embeddings


def retrieve_top_chunks(question: str, embedding_model: str, chunk_embeddings, ollama_api_url: str, top_k: int = 1):
    if embedding_model is None or chunk_embeddings is None or question.strip() == "":
        return []

    question_embedding = get_ollama_embedding(question, embedding_model, ollama_api_url)
    valid_embeddings = []
    valid_indexes = []

    for index, embedding in enumerate(chunk_embeddings):
        if embedding:
            valid_embeddings.append(embedding)
            valid_indexes.append(index)

    if not valid_embeddings:
        return []

    sims = cosine_similarity([question_embedding], valid_embeddings)[0]
    top_idx = sims.argsort()[::-1][:top_k]
    return [valid_indexes[i] for i in top_idx]


def generate_answer(question: str, top_chunks: List[str]) -> str:
    if not top_chunks:
        return "No relevant content found to answer the question."
    # simple heuristic: return the most similar chunk and a short summary
    best = top_chunks[0]
    excerpt = best[:1000]
    return f"Relevant chunk excerpt:\n\n{excerpt}\n\n(Answering is a simulated local response; integrate an LLM for richer answers.)"


def generate_answer_with_llama(
    question: str,
    top_chunks: List[str],
    model: str,
    ollama_api_url: str,
) -> str:
    if not top_chunks:
        return "No relevant content found to answer the question."

    context = "\n\n".join([c for c in top_chunks if c.strip()])[:5000]
    if not context:
        return "No relevant content found to answer the question."

    prompt = (
        "You are a helpful assistant. Answer only from the provided context. "
        "If the answer is not in the context, say you cannot find it in the document.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context}\n\n"
        "Answer clearly in 4-6 lines."
    )

    base_url = ollama_api_url.rstrip("/")
    endpoint = f"{base_url}/api/generate"

    try:
        resp = requests.post(
            endpoint,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2},
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        text = (data.get("response") or "").strip()
        if text:
            return text
        return "Llama returned an empty response."
    except Exception as exc:
        fallback = generate_answer(question, top_chunks)
        return (
            f"Could not reach Llama via Ollama ({exc}). Showing fallback answer.\n\n{fallback}"
        )


# ============================================
# RUN DEMO
# ============================================

with right_col:
    if run_btn:

        if uploaded_pdf is None:
            st.warning("Please upload a PDF first.")
        else:
            # extract and cache document text
            raw_bytes = uploaded_pdf.read()
            # Create a reader from bytes for stats
            try:
                reader = PdfReader(io.BytesIO(raw_bytes))
            except Exception:
                reader = None

            # measure extraction time
            t0 = time.perf_counter()
            if not st.session_state.document_text:
                text = extract_text_from_pdf(raw_bytes)
                st.session_state.document_text = text
            else:
                text = st.session_state.document_text
            t_extract = time.perf_counter() - t0

            # DOCUMENT STATS (dynamic)
            st.subheader("📄 Document Statistics")
            stats = compute_stats(text, reader)
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Pages", stats.get("pages"))
            with c2:
                st.metric("Words", f"{stats.get('words'):,}")
            with c3:
                st.metric("Tokens", f"{stats.get('tokens'):,}")

            st.divider()

            # TIMING: simulated LLM times to illustrate difference
            if not enable_overlap:
                st.error(
                    """
### ❌ No Chunking

Entire document is sent to the LLM.

Problems:

• Context overload

• Token limit issues

• Information loss

• Lower answer quality
"""
                )

                st.subheader("📄 Document Sent To LLM")
                st.text_area("Entire Document", value=(text[:10000] + "...") if text else "(No text extracted)", height=250)

                # simulate LLM processing time proportional to tokens for illustration
                sim_llm = min(5.0, max(0.2, stats.get("tokens", 0) / 50000.0))
                total_no_chunk_time = t_extract + sim_llm
                st.session_state.last_timing = {"extract_time": t_extract, "no_chunk_time": total_no_chunk_time}

                # display timing
                tcol1, tcol2 = st.columns(2)
                with tcol1:
                    st.metric("Extraction Time (s)", f"{t_extract:.3f}")
                with tcol2:
                    st.metric("Estimated No-Chunk LLM Time (s)", f"{total_no_chunk_time:.3f}")

                # CHATBOT ANSWER - Generate answer from full document
                with chatbot_answer_slot:
                    st.divider()
                    st.header("🤖 Chatbot Answer")

                    if user_question.strip() == "":
                        st.info("Type a question and press Run Demo to retrieve an answer.")
                    else:
                        # Send entire document as context
                        with st.spinner("Generating answer from LLM..."):
                            answer = generate_answer_with_llama(
                                user_question,
                                [text],  # Send entire document as single context
                                llama_model,
                                ollama_url,
                            )
                        st.subheader(f"Answer (Llama: {llama_model})")
                        st.write(answer)

            else:
                # CHUNKING MODE
                st.success(f"""
✅ Chunking Completed

Chunks Created: {num_chunks}

Overlap Words: {overlap_words}
""")
                st.info(
                    """
### Why Chunking Helps

• Better Retrieval

• Better Recall

• Better Precision

• Lower Token Usage

• Improved Answers
"""
                )

                st.divider()

                # create chunks and measure
                t_chunk_start = time.perf_counter()
                st.session_state.chunks = create_chunks(text, overlap_words, chunk_strategy)
                retriever_key = (
                    hashlib.md5(text.encode("utf-8")).hexdigest(),
                    chunk_strategy,
                    overlap_words,
                    embedding_model,
                    len(st.session_state.chunks),
                )
                try:
                    if (
                        st.session_state.chunk_embeddings is None
                        or st.session_state.embedding_model != embedding_model
                        or st.session_state.retriever_key != retriever_key
                    ):
                        st.session_state.embedding_model, st.session_state.chunk_embeddings = build_retriever(
                            st.session_state.chunks,
                            embedding_model,
                            ollama_url,
                        )
                        st.session_state.retriever_key = retriever_key
                    else:
                        st.info("Reusing cached chunk embeddings for this document and strategy.")
                except Exception as exc:
                    st.session_state.embedding_model = None
                    st.session_state.chunk_embeddings = None
                    st.session_state.retriever_key = None
                    st.warning(f"Embedding generation failed: {exc}")
                t_chunk = time.perf_counter() - t_chunk_start

                # show chunk dropdown and content
                st.subheader("📄 Created Chunks")
                chunk_names = [f"Chunk {i}" for i in range(1, len(st.session_state.chunks) + 1)]
                selected_chunk = st.selectbox("Select Chunk", chunk_names)
                chunk_number = int(selected_chunk.split(" ")[1]) - 1
                st.text_area("Chunk Content", value=st.session_state.chunks[chunk_number], height=250)

                with st.expander("View All Chunks", expanded=False):
                    for i in range(1, len(st.session_state.chunks) + 1):
                        st.markdown(f"### Chunk {i}")
                        st.write(st.session_state.chunks[i - 1])
                        st.divider()

                if enable_overlap:
                    st.subheader("🔄 Overlap Visualization")
                    st.code(f"""
Overlap Words: {overlap_words}

Showing first two chunks for visualization:

Chunk 1:\n{st.session_state.chunks[0][:200]}\n\nChunk 2:\n{st.session_state.chunks[1][:200] if len(st.session_state.chunks) > 1 else ''}
""")

                # timing display
                total_chunk_time = t_extract + t_chunk
                st.session_state.last_timing = {"extract_time": t_extract, "chunk_time": t_chunk, "total_chunk_time": total_chunk_time}
                tcol1, tcol2 = st.columns(2)
                with tcol1:
                    st.metric("Extraction Time (s)", f"{t_extract:.3f}")
                with tcol2:
                    st.metric("Chunking Time (s)", f"{t_chunk:.3f}")

                # CHATBOT ANSWER (anchored below User Question)
                with chatbot_answer_slot:
                    st.divider()
                    st.header("🤖 Chatbot Answer")

                    if user_question.strip() == "":
                        st.info("Type a question and press Run Demo to retrieve an answer.")
                    else:
                        if st.session_state.embedding_model is None or st.session_state.chunk_embeddings is None:
                            st.info("No chunks to search. Make sure chunking is enabled and Run Demo was pressed.")
                        else:
                            try:
                                retrieval_count = min(num_chunks, len(st.session_state.chunks))
                                top_idxs = retrieve_top_chunks(
                                    user_question,
                                    st.session_state.embedding_model,
                                    st.session_state.chunk_embeddings,
                                    ollama_url,
                                    top_k=retrieval_count,
                                )
                            except Exception as exc:
                                st.error(f"Could not compute embeddings for the question: {exc}")
                                top_idxs = []
                            if not top_idxs:
                                st.write("No relevant chunk found.")
                            else:
                                top_chunks = [st.session_state.chunks[idx] for idx in top_idxs]
                                st.subheader(f"Top {len(top_chunks)} Retrieved Chunks Sent to LLM")
                                for position, chunk_text in enumerate(top_chunks, start=1):
                                    with st.expander(f"Retrieved Chunk {position}", expanded=(position == 1)):
                                        st.write(chunk_text)

                                try:
                                    with st.spinner("Generating answer from LLM..."):
                                        answer = generate_answer_with_llama(
                                            user_question,
                                            top_chunks,
                                            llama_model,
                                            ollama_url,
                                        )
                                    st.subheader(f"Answer (Llama: {llama_model})")
                                    st.write(answer)
                                except Exception as exc:
                                    st.error(f"Error generating answer: {exc}")

    else:
        with chatbot_answer_slot:
            st.header("🤖 Chatbot Answer")
            st.info("""
Steps:

1. Upload PDF

2. Enable Chunking with Overlap (optional)

3. Choose Chunking Strategy

4. Select Top Chunks to Send to LLM

5. Click Run Demo

6. View Generated Chunks

7. Ask Questions

This helps demonstrate to stakeholders
how chunking works behind the scenes.
""")