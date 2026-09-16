"""Veris document workspace. Run: streamlit run app.py"""
import html
import os
from pathlib import Path
import streamlit as st
from veris_core import DocumentError, answer_from_docs, build_corpus, demo_corpus, embed, retrieve

st.set_page_config(page_title="Veris | Document workspace", layout="wide")
st.markdown('<style>' + Path(__file__).with_name('style.css').read_text() + '</style>', unsafe_allow_html=True)
for key, value in {"corpus": None, "history": [], "result": None, "generation": 0}.items():
    if key not in st.session_state:
        st.session_state[key] = value


def replace_workspace(candidate):
    st.session_state.corpus = candidate
    st.session_state.history = []
    st.session_state.result = None
    st.session_state.generation += 1


def submit(question, selected, assisted):
    corpus = st.session_state.corpus
    vector, notice = None, ""
    if assisted and corpus.vectors is not None:
        try:
            vector = embed([question])[0]
        except Exception:
            notice = "Semantic search is unavailable. Results below use local text search."
    hits = retrieve(corpus, question, selected, query_vector=vector)
    answer = ""
    if assisted and hits:
        try:
            answer = answer_from_docs(question, hits)
        except Exception:
            notice = "An assisted answer is unavailable. Read the matching source passages below."
    result = dict(question=question, answer=answer, hits=hits, notice=notice, scope=list(selected))
    st.session_state.result = result
    st.session_state.history.append(result)
    st.session_state.history = st.session_state.history[-30:]


with st.sidebar:
    st.markdown('<div class="brand">veris.</div><div class="eyebrow">Document workspace</div>', unsafe_allow_html=True)
    st.markdown('### Library')
    st.caption('Up to 10 PDFs · 15 MB each · 40 MB total')
    files = st.file_uploader('Add PDF documents', type=['pdf'], accept_multiple_files=True,
                             key=f'uploads_{st.session_state.generation}')
    if st.button('Index documents', type='primary', use_container_width=True, disabled=not files):
        progress = st.progress(0, text='Preparing documents')
        try:
            candidate = build_corpus([(f.name, f.getvalue()) for f in files], progress.progress)
            replace_workspace(candidate)
            st.rerun()
        except DocumentError as exc:
            st.error(str(exc))
            st.caption('Your previous workspace is still available.')
        except Exception:
            st.error('Indexing could not finish. Try a smaller document. Your previous workspace is still available.')
        finally:
            progress.empty()
    st.caption('A new collection replaces the current one only after every PDF succeeds.')
    corpus = st.session_state.corpus
    if corpus:
        names = list(corpus.documents)
        selected = st.multiselect('Search within', names, default=names, key=f'scope_{st.session_state.generation}')
        st.caption(f'{len(names)} documents / {len(corpus.passages)} passages')
        for notice in corpus.notices:
            st.caption(notice)
    else:
        selected = []
    st.divider()
    has_key = bool(os.getenv('OPENAI_API_KEY'))
    mode = st.selectbox('Answer mode', ['Source search', 'Assisted answers'])
    assisted = mode == 'Assisted answers' and has_key
    if mode == 'Assisted answers':
        if not has_key:
            st.info('Assisted answers need an OPENAI_API_KEY on the server. Source search remains available.')
        else:
            st.caption('Questions and matching excerpts are sent to OpenAI. Verify answers against the sources.')
    else:
        st.caption('Search extracted text on this server. No model or API key needed.')
    if corpus and has_key and assisted and corpus.vectors is None:
        st.caption('Optional semantic search sends all indexed passages to OpenAI for embedding.')
        if st.button('Enable semantic search', use_container_width=True):
            with st.spinner('Preparing semantic search'):
                try:
                    vectors = embed([p.text for p in corpus.passages])
                    corpus.vectors = vectors
                    st.success('Semantic search is ready.')
                except Exception:
                    st.warning('Semantic search could not be enabled. Your local index is intact.')
    elif corpus and corpus.vectors is not None:
        st.caption('Semantic index available for assisted mode')
    if st.button('Open sample workspace', use_container_width=True):
        replace_workspace(demo_corpus())
        st.rerun()
    if corpus and st.button('Clear workspace', use_container_width=True):
        replace_workspace(None)
        st.rerun()
    st.markdown('<div class="foot">Session workspace<br>Download useful results before leaving.</div>', unsafe_allow_html=True)

st.markdown('<div class="eyebrow">Read closely. Find what matters.</div>', unsafe_allow_html=True)
st.title('Your documents, within reach.')
st.markdown('<div class="deck">Search the details, compare the evidence, and return to the exact page. A focused place to work with your PDFs.</div>', unsafe_allow_html=True)
workspace, activity, privacy, terms_tab = st.tabs(['Workspace', 'Recent searches', 'Privacy', 'Terms'])

with workspace:
    if not corpus:
        st.markdown('## Start with a document')
        st.write('Add PDFs using the library, then index them to make their text searchable. A document with selectable text works best.')
        st.divider()
        left, right = st.columns([3, 2], gap='large')
        with left:
            st.markdown('### See the workspace in use')
            st.write('The sample event handbook includes registration times, cancellation rules, and volunteer instructions. Search its actual text and inspect the matching pages.')
            if st.button('Try the sample handbook', type='primary'):
                replace_workspace(demo_corpus())
                st.rerun()
        with right:
            st.markdown('<div class="source-label">Sample excerpt / Page 2</div><div class="excerpt">Registration cancellations received at least seven days before the event receive a full refund.</div>', unsafe_allow_html=True)
            st.caption('Fictional sample content. No account or API key required.')
    else:
        with st.form('question_form'):
            question = st.text_input('What would you like to find?', placeholder='Ask a question or enter a phrase', max_chars=2000)
            asked = st.form_submit_button('Find in documents', type='primary', disabled=not selected)
        if not selected:
            st.info('Select at least one document in the library to search.')
        if asked and question.strip() and selected:
            with st.spinner('Finding relevant passages'):
                submit(question.strip(), selected, assisted)
        result = st.session_state.result
        if result and set(result['scope']) != set(selected):
            st.info('Your document selection changed. Search again to refresh the results.')
        elif result:
            hits = result['hits']
            left, right = st.columns([1.15, 1], gap='large')
            with left:
                st.markdown('### Search results')
                st.write(result['question'])
                if result['notice']:
                    st.info(result['notice'])
                if result['answer']:
                    st.markdown(result['answer'])
                    st.caption('Generated from excerpts. Source labels identify passages, not a guarantee of accuracy.')
                elif not hits:
                    st.info('No matching evidence found. Try a name, course code, or exact phrase, or include another document.')
                else:
                    st.caption('Matching passages from your selected documents.')
                for hit in hits:
                    st.markdown(f"**[{hit['id']}] {html.escape(hit['filename'])} · Page {hit['page']}**")
                    st.text(hit['text'][:400] + ('…' if len(hit['text']) > 400 else ''))
                if hits:
                    export = '# ' + result['question'] + '\n\n' + result['answer'] + '\n\n'
                    export += '\n\n'.join(f"[{h['id']}] {h['filename']} | Page {h['page']}\n{h['text']}" for h in hits)
                    st.download_button('Download this result', export, file_name='veris-result.md', mime='text/markdown')
            with right:
                st.markdown('### Source reader')
                if hits:
                    hit_id = st.selectbox('Choose a source', list(range(len(hits))),
                                          format_func=lambda i: f"[{hits[i]['id']}] {hits[i]['filename']} · Page {hits[i]['page']}")
                    hit = hits[hit_id]
                    st.markdown(f'<div class="source-label">{html.escape(hit["filename"])} / Page {hit["page"]}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="excerpt">{html.escape(hit["text"])}</div>', unsafe_allow_html=True)
                    st.caption('Extracted passage. Verify tables and grades in the original PDF.')
                    raw = corpus.documents[hit['filename']]
                    if raw:
                        st.download_button('Download original PDF', raw, file_name=hit['filename'], mime='application/pdf')
                else:
                    st.caption('A matching passage will appear here after your next search.')
        else:
            st.markdown('### Ready to search')
            st.write('Ask about a specific detail. Every result includes its document and page number.')
            st.caption('For transcripts, start with a course name or code. PDF extraction may not preserve table columns.')

with activity:
    st.markdown('## Recent searches')
    st.caption('The latest 30 searches in this session. Starting a new collection clears this history.')
    if not st.session_state.history:
        st.write('Your searches will appear here.')
    for result in reversed(st.session_state.history):
        with st.expander(result['question']):
            st.write(result['answer'] or 'Source search')
            for hit in result['hits']:
                st.caption(f"[{hit['id']}] {hit['filename']} · Page {hit['page']}")
                st.text(hit['text'])

with privacy:
    st.markdown('## Privacy & data handling')
    st.write('Veris processes uploads on the server hosting this app. Source search does not send text to a model provider. PDF bytes and extracted passages are held in the current session; extraction also uses temporary files removed after processing.')
    st.write("Assisted answers send questions and retrieved excerpts to OpenAI. Enabling semantic search separately sends every indexed passage to OpenAI. Provider retention and the hosting operator's logging policies may still apply.")
    st.write("This version does not write a shared document index, add analytics, or use an application database. Clear workspace removes the app's references to the collection and history. It does not guarantee immediate erasure from server memory, host backups, or provider systems.")
    st.write('Upload only documents you are authorized to process. Use a trusted deployment with appropriate access controls for sensitive records. Ask the deployment operator about hosting, retention, and privacy requests before uploading.')

with terms_tab:
    st.markdown('## Terms of use')
    st.write("Use this workspace only with documents you have permission to upload and process. Do not access other users' information or disrupt the service.")
    st.write('Search results and assisted answers can omit context or contain errors. Verify important details against the original, particularly grades, dates, amounts, and tables. This tool is not a substitute for professional advice or an official record.')
    st.write('This self-hosted software has no service availability commitment. Sessions may expire and work may be lost. Download results you need to retain. The deployment operator is responsible for publishing additional terms, contact details, and privacy obligations that apply to their service.')

st.divider()
st.markdown('<div class="foot">VERIS / Document workspace · Source-first by default</div>', unsafe_allow_html=True)
