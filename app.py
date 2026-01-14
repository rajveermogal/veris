# app.py
import os
import re
import streamlit as st
from typing import Optional, Tuple, List, Dict, Any

from veris_core import (
    pdf_bytes_to_passages,
    build_store,
    search,
    make_context,
    answer_from_docs,
)

from openai import OpenAI

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
RETRIEVE_K = 24

BASE_CONF = 0.22
ENTITY_CONF = 0.10
SUMMARY_CONF = 0.18
GENERAL_FALLBACK_CONF = 0.16

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

# -------------------------------------------------
# Helpers
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


def _safe_openai_text(prompt: str, model: str = "gpt-4.1-mini") -> str:
    try:
        resp = client.responses.create(model=model, input=prompt)
        if hasattr(resp, "output_text") and resp.output_text:
            return resp.output_text.strip()
        try:
            return resp.output[0].content[0].text.strip()  # type: ignore
        except Exception:
            return str(resp).strip()
    except Exception:
        try:
            cc = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            return (cc.choices[0].message.content or "").strip()
        except Exception as e:
            return f"(General knowledge temporarily unavailable: {e})"


def detect_doc_type_from_meta(meta: List[Dict[str, Any]]) -> str:
    if not meta:
        return "unknown"

    sample = meta[:DOC_CLASSIFY_PASSAGES]
    sample_text = "\n\n".join(
        f"[{m.get('filename','?')} p.{m.get('page','?')}] {str(m.get('text',''))[:400]}"
        for m in sample
    )

    prompt = (
        "Classify the primary document type from the excerpted passages.\n"
        "Return EXACTLY one label from this list:\n"
        "- syllabus\n"
        "- policy\n"
        "- technical_manual\n"
        "- automotive\n"
        "- weather\n"
        "- finance\n"
        "- academic_paper\n"
        "- other\n\n"
        "Rules:\n"
        "- Choose the single best label.\n"
        "- Output ONLY the label.\n\n"
        f"Passages:\n{sample_text}"
    )

    label = _safe_openai_text(prompt).strip().lower()
    allowed = {"syllabus", "policy", "technical_manual", "automotive", "weather", "finance", "academic_paper", "other"}
    return label if label in allowed else "other"


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


def extract_numbers_table(hits: List[Dict[str, Any]], max_rows: int = 12) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    num_re = re.compile(r"(-?\d+(?:\.\d+)?)")
    unit_re = re.compile(r"\b(mph|km/h|kph|km|mi|mpg|°c|°f|c|f|kts|kt|knots|mb|hpa|psi|bar|mm|cm|in|inch|inches|%|\$)\b", re.IGNORECASE)

    for h in hits:
        text = (h.get("text") or "").strip()
        if not text:
            continue

        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue

            nums = num_re.findall(line)
            if not nums:
                continue

            unit = ""
            um = unit_re.search(line)
            if um:
                unit = um.group(0)

            val = ", ".join(nums[:2])

            rows.append({
                "Value(s)": val,
                "Unit": unit,
                "Context": (line[:160] + ("…" if len(line) > 160 else "")),
                "Source": f"{h.get('filename','?')} p.{h.get('page','?')}"
            })

            if len(rows) >= max_rows:
                return rows

    return rows


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
                if 4 <= len(cleaned) <= 140:
                    items.append(cleaned)

            if len(items) >= max_items:
                return items

    return items


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def extract_single_numeric_answer(query: str, hits: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    Doc-agnostic single-value extraction:
    - Find best matching line that mentions the user concept (wind speed, mpg, pressure, etc.)
    - Extract number + unit from that line
    """
    q = _normalize(query)

    tokens = [t for t in re.findall(r"[a-z]+", q) if t not in {"the", "a", "an", "of", "is", "what", "tell", "me", "about"}]
    target_terms: List[str] = []

    for i in range(len(tokens) - 1):
        target_terms.append(tokens[i] + " " + tokens[i + 1])

    target_terms.extend(tokens[:6])

    seen = set()
    target_terms = [t for t in target_terms if not (t in seen or seen.add(t))]

    num_re = re.compile(r"(-?\d+(?:\.\d+)?)")
    unit_re = re.compile(
        r"\b(mph|km/h|kph|km|mi|mpg|°c|°f|c|f|kts|kt|knots|mb|hpa|psi|bar|mm|cm|in|inch|inches|%|\$)\b",
        re.IGNORECASE,
    )

    best_value: Optional[str] = None
    best_hit: Optional[Dict[str, Any]] = None
    best_score = -1

    for h in hits:
        text = (h.get("text") or "").strip()
        if not text:
            continue

        for line in text.splitlines():
            l = line.strip()
            if not l:
                continue

            ln = _normalize(l)

            nums = num_re.findall(ln)
            if not nums:
                continue

            term_score = 0
            for t in target_terms:
                if t and t in ln:
                    term_score += min(6, 1 + len(t.split()))

            unit = ""
            um = unit_re.search(ln)
            if um:
                unit = um.group(0)

            score = term_score + (2 if unit else 0)

            if score > best_score:
                val = ", ".join(nums[:2])
                best_value = f"{val} {unit}".strip()
                best_hit = h
                best_score = score

    if best_value is None or best_score < 2:
        return None, None

    return best_value, best_hit


def answer_general_knowledge(query: str) -> str:
    prompt = (
        "Answer the user question briefly and helpfully.\n"
        "If the question could be location/time dependent, say so.\n"
        "Do NOT mention PDFs. Do NOT fabricate citations.\n\n"
        f"Question: {query}"
    )
    ans = _safe_openai_text(prompt)
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
# Sidebar — Knowledge Base (Clear Chat at bottom)
# -------------------------------------------------
with st.sidebar:
    st.header("Knowledge Base")
    st.caption("Upload PDFs. Veris answers from these documents first.")

    files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

    if st.button("Index documents", type="primary", use_container_width=True):
        if not files:
            st.warning("Upload at least one PDF.")
        else:
            passages: List[Dict[str, Any]] = []
            st.session_state.doc_names = [f.name for f in files]

            with st.spinner("Indexing documents..."):
                for f in files:
                    pdf_bytes = f.read()
                    passages.extend(pdf_bytes_to_passages(pdf_bytes, f.name))

            if not passages:
                st.error("No text could be extracted from the PDFs.")
            else:
                st.session_state.index, st.session_state.meta = build_store(passages)
                st.session_state.indexed = True

                with st.spinner("Detecting document type..."):
                    st.session_state.doc_type = detect_doc_type_from_meta(st.session_state.meta)

                st.success(f"Indexed {len(files)} document(s).")

    st.divider()

    if st.session_state.indexed:
        st.success("Status: Indexed ✅")
        st.caption(doc_mode_hint(st.session_state.doc_type))
        if st.session_state.doc_names:
            st.caption("Active documents:")
            for name in st.session_state.doc_names:
                st.write(f"✅ {name}")
    else:
        st.info("Status: Not indexed")

    st.caption("Veris cites sources when answering from documents.")

    st.markdown("---")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.chat = []
        st.rerun()

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
                docs = ", ".join(st.session_state.doc_names) if st.session_state.doc_names else "your indexed documents"
                reply = f"Hi 👋\n\nAsk me anything about **{docs}** — I’ll cite sources when it’s from your PDFs."
            else:
                reply = "Hi 👋\n\nUpload a PDF and click **Index documents** to get started."
            st.markdown(reply)
            st.session_state.chat.append({"role": "assistant", "content": reply})

        elif not st.session_state.indexed:
            reply = answer_general_knowledge(query)
            st.markdown(reply)
            st.session_state.chat.append({"role": "assistant", "content": reply})

        else:
            with st.spinner("Searching documents..."):
                hits_all = search(st.session_state.index, st.session_state.meta, query, k=RETRIEVE_K)

            top_score = hits_all[0]["score"] if hits_all else 0.0
            effective_conf = ENTITY_CONF if (is_entity_question(query) or looks_like_numbers_request(query)) else BASE_CONF

            ql = query.strip().lower()

            if any(x in ql for x in ["teaching team", "teaching staff", "course staff", "staff list"]):
                team, team_hit = extract_teaching_team(hits_all)
                if team and team_hit:
                    reply = f"{team}\n\nSource: {team_hit['filename']} p.{team_hit['page']}"
                else:
                    reply = "I couldn’t find a clearly labeled staff/team section in the retrieved text."
                st.markdown(reply)
                st.session_state.chat.append({"role": "assistant", "content": reply})

            elif is_entity_question(query) or any(x in ql for x in ["ta", "teaching assistant", "grader", "instructor", "professor", "email", "phone", "office hours"]):
                extracted, src_hit = extract_labeled_field_from_hits(query, hits_all)
                if extracted and src_hit:
                    reply = f"**Answer:** {extracted}\n\nSource: {src_hit['filename']} p.{src_hit['page']}"
                else:
                    if hits_all and hits_all[0]["score"] >= effective_conf:
                        context, _ = make_context(hits_all[:SHOW_K])
                        with st.spinner("Formulating answer..."):
                            reply = answer_from_docs(query, context)
                    else:
                        reply = answer_general_knowledge(query)
                st.markdown(reply)
                st.session_state.chat.append({"role": "assistant", "content": reply})

            elif looks_like_numbers_request(query):
                if hits_all and hits_all[0]["score"] >= effective_conf:
                    single, single_hit = extract_single_numeric_answer(query, hits_all[:SHOW_K])

                    if single and single_hit:
                        reply = f"**Answer:** {single}\n\nSource: {single_hit['filename']} p.{single_hit['page']}"
                        st.markdown(reply)
                        st.session_state.chat.append({"role": "assistant", "content": reply})
                    else:
                        rows = extract_numbers_table(hits_all[:SHOW_K], max_rows=12)
                        if rows:
                            st.markdown("**Extracted values from your PDFs:**")
                            st.table(rows)
                            reply = "Ask a more specific target like: “max wind” / “wind speed at 500mb” / “mpg for model X”."
                            st.markdown(reply)
                            st.session_state.chat.append({"role": "assistant", "content": reply})
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

            elif looks_like_list_request(query):
                if hits_all and hits_all[0]["score"] >= effective_conf:
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
                if hits_all and hits_all[0]["score"] >= BASE_CONF:
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
                if not hits_all or hits_all[0]["score"] < SUMMARY_CONF:
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
                if hits_all and hits_all[0]["score"] >= BASE_CONF:
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
