# app.py
import os
import re
import json
import time
import streamlit as st
from typing import Optional, Tuple, List, Dict, Any

from openai import OpenAI

from veris_core import (
    Passage,
    pdf_bytes_to_passages,
    build_store,
    load_store,
    save_store,
    search_hybrid,
    make_context,
    answer_from_docs,
    detect_doc_type_from_meta,
    sha256_bytes,
)

# -------------------------------------------------
# Branding
# -------------------------------------------------
st.set_page_config(page_title="Veris", page_icon="🐦‍🔥", layout="wide")
st.title("🐦‍🔥 Veris")
st.caption("Understands your documents, so you don’t have to.")

if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY not set. Set it and restart the terminal.")
    st.stop()

client = OpenAI()

# -------------------------------------------------
# Constants
# -------------------------------------------------
SHOW_K = 6
VEC_K = 28
BM25_K = 28
FINAL_K = 24
RERANK_K = 12

BASE_MIN_CONF = 0.22
SUMMARY_MIN_CONF = 0.18
ENTITY_MIN_CONF = 0.12

DOC_CLASSIFY_PASSAGES = 16

# -------------------------------------------------
# Session State
# -------------------------------------------------
if "index" not in st.session_state:
    st.session_state.index = None
if "meta" not in st.session_state:
    st.session_state.meta = []
if "chat" not in st.session_state:
    st.session_state.chat = []
if "indexed" not in st.session_state:
    st.session_state.indexed = False
if "doc_names" not in st.session_state:
    st.session_state.doc_names = []
if "doc_type" not in st.session_state:
    st.session_state.doc_type = "unknown"
if "store_dir" not in st.session_state:
    st.session_state.store_dir = "./veris_store"
if "active_docs" not in st.session_state:
    st.session_state.active_docs = []
if "kb_fingerprints" not in st.session_state:
    st.session_state.kb_fingerprints = []

# -------------------------------------------------
# Intent helpers
# -------------------------------------------------
def is_greeting(text: str) -> bool:
    t = (text or "").strip().lower()
    return t in {"hi", "hello", "hey", "yo", "good morning", "good afternoon", "good evening"}

def is_summary_request(text: str) -> bool:
    t = (text or "").strip().lower()
    return any(x in t for x in ["summary", "summarize", "important points", "key points", "highlights", "important data"])

def is_compare_request(text: str) -> bool:
    t = (text or "").strip().lower()
    return ("compare" in t) or (" vs " in t) or ("versus" in t)

def is_entity_question(text: str) -> bool:
    t = (text or "").strip().lower()
    return any(x in t for x in ["who is", "who's", "who’s", "name of", "email of", "contact for", "phone", "address"])

def looks_like_list_request(text: str) -> bool:
    t = (text or "").strip().lower()
    triggers = ["list", "what are the", "which are", "types of", "models", "parts", "features", "requirements", "steps"]
    return any(x in t for x in triggers)

def looks_like_numbers_request(text: str) -> bool:
    t = (text or "").strip().lower()
    triggers = ["how much", "how many", "price", "cost", "mileage", "mpg", "km", "mph", "temperature", "wind", "speed", "pressure", "mm", "inches"]
    return any(x in t for x in triggers)

def answer_general_knowledge(query: str) -> str:
    prompt = (
        "Answer the user question briefly and helpfully.\n"
        "If the question could be location/time dependent, say so.\n"
        "Do NOT mention PDFs.\n\n"
        f"Question: {query}"
    )
    try:
        resp = client.responses.create(model="gpt-4.1-mini", input=prompt)
        ans = (resp.output_text or "").strip()
    except Exception as e:
        ans = f"(General knowledge temporarily unavailable: {e})"
    return f"**General knowledge (not from your PDFs):**\n\n{ans}"

def doc_mode_hint(doc_type: str) -> str:
    mapping = {
        "syllabus": "Detected type: **Syllabus**",
        "policy": "Detected type: **Policy**",
        "technical_manual": "Detected type: **Technical manual**",
        "automotive": "Detected type: **Automotive / vehicle**",
        "weather": "Detected type: **Weather / meteorological**",
        "finance": "Detected type: **Finance / numbers**",
        "academic_paper": "Detected type: **Academic paper**",
        "other": "Detected type: **Other**",
        "unknown": "Detected type: **Unknown**",
    }
    return mapping.get(doc_type, "Detected type: **Other**")

# -------------------------------------------------
# Extractors (from your original + safe)
# -------------------------------------------------
def _labels_for_query(query: str) -> Tuple[List[str], List[str]]:
    q = (query or "").lower()

    if any(x in q for x in ["ta", "teaching assistant"]):
        targets = ["Teaching Assistant", "TA"]
    elif any(x in q for x in ["professor", "instructor", "teacher"]):
        targets = ["Instructor", "Professor"]
    elif "grader" in q:
        targets = ["Grader"]
    elif any(x in q for x in ["email", "e-mail"]):
        targets = ["Email", "E-mail"]
    elif "phone" in q:
        targets = ["Phone", "Telephone"]
    elif "office hours" in q:
        targets = ["Office Hours"]
    elif "website" in q:
        targets = ["Website", "Class Website", "Course Website"]
    else:
        targets = []

    stops = [
        "Instructor", "Professor", "Teaching Assistant", "TA", "Grader",
        "Office", "Email", "E-mail", "Phone", "Telephone",
        "Website", "Class Website", "Course Website",
        "Course Administrator", "Administrator",
        "Office Hours", "Location", "Zoom", "Canvas", "Slack"
    ]

    return targets, stops

def extract_labeled_field_from_hits(query: str, hits: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    targets, stops = _labels_for_query(query)
    if not targets:
        return None, None

    target_re = re.compile(
        r"(?:" + "|".join(re.escape(t) for t in targets) + r")\s*[:\-]\s*(.+)",
        flags=re.IGNORECASE
    )

    stop_re = re.compile(
        r"\b(?:" + "|".join(re.escape(s) for s in stops) + r")\s*[:\-]",
        flags=re.IGNORECASE
    )

    email_re = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", flags=re.IGNORECASE)

    for h in hits:
        text = (h.get("text") or "").strip()
        if not text:
            continue

        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue

            m = target_re.search(line)
            if not m:
                continue

            value = m.group(1).strip()
            cut = stop_re.search(value)
            if cut:
                value = value[:cut.start()].strip()

            value = re.sub(r"\s{2,}", " ", value).strip(" •-–—:;,. ")

            em = email_re.search(value)
            if em:
                email = em.group(0)
                name_part = value[:em.start()].strip(" ,;:-–—")
                if name_part:
                    return f"{name_part} — {email}", h
                return email, h

            return (value if value else None), h

    return None, None

def extract_teaching_team(hits: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    def find(label_variants: List[str]) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        pat = re.compile(r"(?:" + "|".join(re.escape(x) for x in label_variants) + r")\s*[:\-]\s*(.+)", re.IGNORECASE)
        stop_pat = re.compile(
            r"\b(?:Instructor|Professor|Teaching Assistant|TA|Grader|Office|Email|E-mail|Phone|Website|Office Hours|Location)\s*[:\-]",
            re.IGNORECASE
        )
        email_pat = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)

        for h in hits:
            for line in (h.get("text") or "").splitlines():
                line = line.strip()
                m = pat.search(line)
                if not m:
                    continue
                value = m.group(1).strip()
                cut = stop_pat.search(value)
                if cut:
                    value = value[:cut.start()].strip()
                value = re.sub(r"\s{2,}", " ", value).strip(" •-–—:;,. ")

                em = email_pat.search(value)
                if em:
                    email = em.group(0)
                    name_part = value[:em.start()].strip(" ,;:-–—")
                    if name_part:
                        return f"{name_part} — {email}", h
                    return email, h

                return value, h

        return None, None

    prof, prof_hit = find(["Instructor", "Professor"])
    ta, ta_hit = find(["Teaching Assistant", "TA"])
    grader, grader_hit = find(["Grader"])

    parts = []
    best_hit = prof_hit or ta_hit or grader_hit

    if prof:
        parts.append(f"**Instructor/Professor:** {prof}")
    if ta:
        parts.append(f"**Teaching Assistant:** {ta}")
    if grader:
        parts.append(f"**Grader:** {grader}")

    if not parts:
        return None, None

    return "\n".join(parts), best_hit

def extract_bulleted_list(hits: List[Dict[str, Any]], max_items: int = 10) -> List[str]:
    items: List[str] = []
    for h in hits:
        text = (h.get("text") or "").strip()
        if not text:
            continue

        for line in text.splitlines():
            s = line.strip()
            if not s:
                continue

            if s.startswith(("-", "•", "*")) or re.match(r"^\d+[\.\)]\s+", s):
                cleaned = re.sub(r"^\d+[\.\)]\s+", "", s).lstrip("-•* ").strip()
                cleaned = re.sub(r"\s{2,}", " ", cleaned)
                if 4 <= len(cleaned) <= 160:
                    items.append(cleaned)

            if len(items) >= max_items:
                return items

    return items

# -------------------------------------------------
# Confidence heuristic
# -------------------------------------------------
def confident_enough(hits: List[Dict[str, Any]], min_conf: float) -> bool:
    if not hits:
        return False
    top = float(hits[0].get("score", 0.0))
    second = float(hits[1].get("score", 0.0)) if len(hits) > 1 else 0.0
    margin = top - second
    if margin < 0.03 and top < (min_conf + 0.05):
        return False
    return top >= min_conf

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
with st.sidebar:
    st.header("Knowledge Base")
    st.caption("Upload PDFs. Veris answers from these documents first.")

    st.session_state.store_dir = st.text_input(
        "Store folder (for persistence)",
        value=st.session_state.store_dir,
        help="Veris will save/load indexes here so you don't re-index every time.",
    )

    files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

    colA, colB = st.columns(2)
    with colA:
        do_index = st.button("Index / Update", type="primary", use_container_width=True)
    with colB:
        do_load = st.button("Load from disk", use_container_width=True)

    if do_load:
        try:
            idx, meta, kb = load_store(st.session_state.store_dir)
            st.session_state.index = idx
            st.session_state.meta = meta
            st.session_state.indexed = True
            st.session_state.doc_names = sorted({m.get("filename", "?") for m in meta})
            st.session_state.active_docs = list(st.session_state.doc_names)
            st.session_state.kb_fingerprints = kb.get("fingerprints", [])
            st.session_state.doc_type = detect_doc_type_from_meta(meta[:DOC_CLASSIFY_PASSAGES])
            st.success(f"Loaded store from {st.session_state.store_dir}")
        except Exception as e:
            st.error(f"Load failed: {e}")

    if do_index:
        if not files:
            st.warning("Upload at least one PDF.")
        else:
            passages: List[Passage] = []
            st.session_state.doc_names = [f.name for f in files]
            fingerprints: List[str] = []

            with st.spinner("Reading PDFs..."):
                for f in files:
                    pdf_bytes = f.read()
                    fingerprints.append(sha256_bytes(pdf_bytes))
                    passages.extend(pdf_bytes_to_passages(pdf_bytes, f.name))

            if not passages:
                st.error("No text could be extracted from the PDFs.")
            else:
                with st.spinner("Building index (cached embeddings + hybrid prep)..."):
                    idx, meta = build_store(passages, store_dir=st.session_state.store_dir)

                st.session_state.index = idx
                st.session_state.meta = meta
                st.session_state.indexed = True
                st.session_state.kb_fingerprints = fingerprints

                with st.spinner("Detecting document type..."):
                    st.session_state.doc_type = detect_doc_type_from_meta(st.session_state.meta[:DOC_CLASSIFY_PASSAGES])

                with st.spinner("Saving store..."):
                    save_store(
                        st.session_state.store_dir,
                        st.session_state.index,
                        st.session_state.meta,
                        kb_info={"fingerprints": fingerprints, "doc_names": st.session_state.doc_names, "saved_at": time.time()},
                    )

                st.session_state.active_docs = list(st.session_state.doc_names)
                st.success(f"Indexed {len(files)} document(s) and saved store.")

    st.divider()

    if st.session_state.indexed:
        st.success("Status: Indexed ✅")
        st.caption(doc_mode_hint(st.session_state.doc_type))
        if st.session_state.doc_names:
            st.caption("Active documents:")
            st.session_state.active_docs = st.multiselect(
                "Search scope",
                options=st.session_state.doc_names,
                default=st.session_state.active_docs if st.session_state.active_docs else st.session_state.doc_names,
                help="Restrict searches to selected PDFs.",
            )
            for name in st.session_state.active_docs:
                st.write(f"✅ {name}")
    else:
        st.info("Status: Not indexed")

    st.caption("Veris cites sources when answering from documents.")
    st.markdown("---")

    if st.button("Clear chat", use_container_width=True):
        st.session_state.chat = []
        st.rerun()

    if st.session_state.chat:
        export = {"chat": st.session_state.chat, "doc_names": st.session_state.doc_names, "active_docs": st.session_state.active_docs}
        st.download_button(
            "Download chat (JSON)",
            data=json.dumps(export, indent=2),
            file_name="veris_chat.json",
            mime="application/json",
            use_container_width=True,
        )

# -------------------------------------------------
# Chat History
# -------------------------------------------------
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------------------------------
# Chat Input
# -------------------------------------------------
query = st.chat_input("Ask Veris about your documents...")

if query:
    st.session_state.chat.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        if is_greeting(query):
            if st.session_state.indexed:
                docs = ", ".join(st.session_state.active_docs) if st.session_state.active_docs else "your indexed documents"
                reply = f"Hi 👋\n\nAsk me anything about **{docs}** — I’ll cite sources when it’s from your PDFs."
            else:
                reply = "Hi 👋\n\nUpload a PDF and click **Index / Update** to get started."
            st.markdown(reply)
            st.session_state.chat.append({"role": "assistant", "content": reply})

        elif not st.session_state.indexed:
            reply = answer_general_knowledge(query)
            st.markdown(reply)
            st.session_state.chat.append({"role": "assistant", "content": reply})

        else:
            with st.spinner("Searching documents (hybrid + rerank)..."):
                hits_all = search_hybrid(
                    index=st.session_state.index,
                    meta=st.session_state.meta,
                    query=query,
                    vec_k=VEC_K,
                    bm25_k=BM25_K,
                    final_k=FINAL_K,
                    rerank_k=RERANK_K,
                    restrict_filenames=st.session_state.active_docs if st.session_state.active_docs else None,
                )

            ql = query.strip().lower()
            is_entity = is_entity_question(query) or any(x in ql for x in ["ta", "teaching assistant", "grader", "instructor", "professor", "email", "phone", "office hours"])
            is_numbers = looks_like_numbers_request(query)
            min_conf = ENTITY_MIN_CONF if (is_entity or is_numbers) else BASE_MIN_CONF

            if any(x in ql for x in ["teaching team", "teaching staff", "course staff", "staff list"]):
                team, team_hit = extract_teaching_team(hits_all)
                if team and team_hit:
                    reply = f"{team}\n\nSource: {team_hit['filename']} p.{team_hit['page']}"
                else:
                    reply = "I couldn’t find a clearly labeled staff/team section in the retrieved text."
                st.markdown(reply)
                st.session_state.chat.append({"role": "assistant", "content": reply})

            elif is_entity:
                extracted, src_hit = extract_labeled_field_from_hits(query, hits_all[:SHOW_K])
                if extracted and src_hit:
                    reply = f"**Answer:** {extracted}\n\nSource: {src_hit['filename']} p.{src_hit['page']}"
                else:
                    if confident_enough(hits_all, min_conf):
                        context, _ = make_context(hits_all[:SHOW_K])
                        with st.spinner("Formulating answer..."):
                            reply = answer_from_docs(query, context)
                    else:
                        reply = answer_general_knowledge(query)
                st.markdown(reply)
                st.session_state.chat.append({"role": "assistant", "content": reply})

            elif looks_like_list_request(query):
                if confident_enough(hits_all, min_conf):
                    items = extract_bulleted_list(hits_all[:SHOW_K], max_items=10)
                    if items:
                        st.markdown("**From your PDFs:**")
                        for it in items:
                            st.markdown(f"- {it}")
                        reply = "If you want, tell me a section/page to focus on and I’ll make it more precise."
                    else:
                        context, _ = make_context(hits_all[:SHOW_K])
                        with st.spinner("Formulating answer..."):
                            reply = answer_from_docs(query, context)
                    st.markdown(reply)
                    st.session_state.chat.append({"role": "assistant", "content": reply})
                else:
                    reply = answer_general_knowledge(query)
                    st.markdown(reply)
                    st.session_state.chat.append({"role": "assistant", "content": reply})

            elif is_compare_request(query):
                if confident_enough(hits_all, BASE_MIN_CONF):
                    context, _ = make_context(hits_all[:SHOW_K])
                    with st.spinner("Comparing using your PDFs..."):
                        reply = answer_from_docs(
                            "Compare the items in the question. Use only the provided context and cite sources.\n\nQuestion: " + query,
                            context
                        )
                else:
                    reply = answer_general_knowledge(query)
                st.markdown(reply)
                st.session_state.chat.append({"role": "assistant", "content": reply})

            elif is_summary_request(query):
                if not confident_enough(hits_all, SUMMARY_MIN_CONF):
                    if st.session_state.doc_type == "weather":
                        reply = (
                            "This document looks like mostly structured meteorological data.\n\n"
                            "To summarize well, tell me what you care about:\n"
                            "- wind speed\n"
                            "- temperature\n"
                            "- pressure\n"
                            "- precipitation\n"
                            "- date range\n\n"
                            "Or ask: “Summarize the first 3 pages.”"
                        )
                    else:
                        reply = (
                            "I can summarize, but I’m not confident I pulled the right section.\n\n"
                            "Try:\n"
                            "- “Summarize page 1”\n"
                            "- “Summarize the section about X”\n"
                            "- Or ask a specific question (e.g., “What are the requirements?”)"
                        )
                else:
                    context, _ = make_context(hits_all[:SHOW_K])
                    with st.spinner("Summarizing from your PDFs..."):
                        reply = answer_from_docs(query, context)

                st.markdown(reply)
                st.session_state.chat.append({"role": "assistant", "content": reply})

            else:
                if confident_enough(hits_all, min_conf):
                    context, _ = make_context(hits_all[:SHOW_K])
                    with st.spinner("Formulating answer..."):
                        reply = answer_from_docs(query, context)
                else:
                    reply = answer_general_knowledge(query)

                st.markdown(reply)
                st.session_state.chat.append({"role": "assistant", "content": reply})

            if hits_all:
                with st.expander("Sources"):
                    for i, h in enumerate(hits_all[:SHOW_K], 1):
                        st.markdown(f"**{i}. {h['filename']} p.{h['page']}** — score `{h['score']:.3f}`")
                        st.write(h["text"])

st.divider()
st.caption("© 2026 Rajveer Mogal. All rights reserved.")
