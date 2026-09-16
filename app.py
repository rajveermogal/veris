"""Veris conversational document workspace."""
import hashlib
import html
import json
import logging
import os
from pathlib import Path
import streamlit as st
from veris_core import build_corpus, DocumentError, demo_corpus
from chat_engine import gather_context, respond

st.set_page_config(page_title='Veris', layout='centered', initial_sidebar_state='expanded')
st.markdown('<style>'+Path(__file__).with_name('style.css').read_text()+'</style>', unsafe_allow_html=True)
for k,v in dict(corpus=None, messages=[], cache={}, generation=0, screen='Chat', calls=0).items():
    if k not in st.session_state: st.session_state[k]=v
if st.session_state.get('citation_version') != 3:
    st.session_state.cache={}
    st.session_state.citation_version=3

def replace(candidate):
    st.session_state.corpus=candidate
    st.session_state.messages=[]
    st.session_state.cache={}
    st.session_state.generation+=1
    st.session_state.screen='Chat'

def upload(location='sidebar'):
    files=st.file_uploader('Choose PDFs',type=['pdf'],accept_multiple_files=True,key=f'files_{location}_{st.session_state.generation}')
    st.caption('15 MB per file. Up to 10 PDFs, 40 MB combined.')
    if st.button('Add to workspace',key='add_'+location,type='primary',disabled=not files,use_container_width=True):
        try:
            with st.spinner('Reading your documents…'):
                existing=list(st.session_state.corpus.documents.items()) if st.session_state.corpus else []
                incoming=[(f.name,f.getvalue()) for f in files]
                incoming_names={n for n,_ in incoming}
                candidate=build_corpus([(n,b) for n,b in existing if b and n not in incoming_names]+incoming)
            replace(candidate)
            st.rerun()
        except DocumentError as exc: st.error(str(exc))
        except Exception: st.error('I could not read these documents. Your existing workspace is unchanged. Try a smaller PDF.')

with st.sidebar:
    st.markdown('<div class="brand">veris<span>.</span></div>',unsafe_allow_html=True)
    for page in ['Chat','Documents','Help']:
        if st.button(page,key='nav_'+page,type='primary' if st.session_state.screen==page else 'secondary',use_container_width=True):
            st.session_state.screen=page
            st.rerun()
    st.divider()
    with st.popover('Add documents',use_container_width=True): upload()
    corpus=st.session_state.corpus
    names=list(corpus.documents) if corpus else []
    if names:
        selected=st.multiselect('In this conversation',names,default=names,key=f'selection_{st.session_state.generation}')
        if st.button('New conversation',use_container_width=True):
            st.session_state.messages=[]
            st.rerun()
    else: selected=[]
    st.caption('Your documents. A little more clarity.')

screen=st.session_state.screen
if screen=='Help':
    st.title('Help & settings')
    with st.expander('How Veris works',expanded=True):
        st.write('Add PDFs, then ask a question in Chat. Follow up naturally. Open Sources below an answer to check the original passage and page. New conversation clears the conversation without removing your documents.')
        st.write('Scanned PDFs need OCR before upload. Tables and complex layouts may lose structure during extraction. Verify important numbers in the original document.')
        if st.button('Load a sample document'):
            replace(demo_corpus());st.rerun()
    with st.expander('Connection & usage'):
        st.write('Connected to OpenAI.' if os.getenv('OPENAI_API_KEY') else 'Add OPENAI_API_KEY in your Streamlit app secrets to enable conversational answers.')
        st.caption('Uploads and retrieval use no model calls. Each new uncached answer uses at most one compact model request. Recent conversation is limited to three turns. Repeating the same request with the same context reuses its answer.')
        st.write(f'Model requests attempted this session: {st.session_state.calls}')
        st.caption('Default model: gpt-4.1-mini. A VERIS_CHAT_MODEL server setting can override it. Provider charges apply; this is not a spending cap across users.')
    with st.expander('Privacy'):
        st.write('Documents are processed on this hosting server. When you ask a question, selected source text, the question and up to three recent conversation turns are sent to OpenAI. Small documents may be included in full. Provider and hosting retention policies apply.')
        st.write('The workspace and answer cache are held in your session. PDF extraction uses temporary files removed after processing. Clearing the workspace removes app references, but cannot guarantee erasure from hosting memory, logs or provider systems. Upload only documents you have permission to process.')
    with st.expander('Terms'):
        st.write('Verify important answers against their sources. Extraction and generated answers may be incomplete or incorrect. The service has no availability guarantee. The deployment operator is responsible for appropriate access controls, contact details and any additional applicable policies.')
    st.stop()
if screen=='Documents':
    st.title('Your documents')
    if corpus:
        with st.popover('Add documents here'): upload('documents_more')
    if not corpus: upload('documents')
    else:
        for name,raw in corpus.documents.items():
            with st.container(border=True):
                st.write(name)
                if raw: st.download_button('Download original',raw,file_name=name,mime='application/pdf',key='download_'+name)
        for notice in corpus.notices: st.caption(notice)
        with st.expander('Remove all documents'):
            st.caption('This also clears the conversation and cached answers.')
            if st.button('Clear workspace'):
                replace(None);st.rerun()
    st.stop()

if not st.session_state.messages:
    st.markdown('<div class="welcome"><div class="intro-label">A SPACE TO THINK CLEARLY</div><h1>What would you like<br>to understand?</h1><p>Add a document. Ask a question. We’ll find it together.</p></div>',unsafe_allow_html=True)
    if not corpus:
        with st.container(key='welcome_upload'):
            if st.button('Add your first document',type='primary',use_container_width=True):
                st.session_state.screen='Documents'
                st.rerun()
else:
    st.markdown('<div class="chat-title">Conversation</div>',unsafe_allow_html=True)

has_key=bool(os.getenv('OPENAI_API_KEY'))
if corpus and not has_key:
    st.info('Chat needs an OpenAI connection. Add OPENAI_API_KEY in Streamlit app settings → Secrets. Your documents are ready.')
if corpus and not selected: st.info('Select a document in the sidebar to continue.')

for i,msg in enumerate(st.session_state.messages):
    with st.chat_message('user'):
        st.markdown('<span class="message-role user-role">You</span>',unsafe_allow_html=True)
        st.write(msg['question'])
    with st.chat_message('assistant'):
        st.markdown('<span class="message-role assistant-role">Veris</span>',unsafe_allow_html=True)
        st.markdown(msg['answer'])
        if msg.get('sources'):
            with st.expander('Sources · '+str(len(msg['sources']))):
                for source in msg['sources']:
                    st.caption(f"{source['filename']} · Page {source['page']}")
                    if source.get('excerpts'):
                        for quote in source['excerpts']:
                            # PDF layout newlines are not paragraph boundaries.
                            preview=' '.join(quote.split())
                            st.markdown('<div class="source-excerpt">'+html.escape(preview)+'</div>',unsafe_allow_html=True)
                    else: st.caption('This earlier answer has page references only. Ask again for precise excerpts.')
        if msg.get('error'): st.caption('No answer was cached. You can submit the question again.')

question=st.chat_input('Ask about your documents…',disabled=not selected or not has_key,max_chars=2000)
if question:
    # Emit the submitted message before retrieval or any blocking API work.
    with st.chat_message('user'):
        st.markdown('<span class="message-role user-role">You</span>',unsafe_allow_html=True)
        st.write(question)
    history=[m for m in st.session_state.messages if not m.get('error') and m.get('scope')==sorted(selected)]
    evidence=gather_context(corpus,question,selected,history)
    key=hashlib.sha256(json.dumps([question,evidence,[(m['question'],m['answer']) for m in history[-3:]]],sort_keys=True).encode()).hexdigest()
    # An immediate retry of the identical question reuses its previous result.
    previous=history[-1] if history else None
    with st.chat_message('assistant'), st.spinner('Reading your documents…'):
        st.markdown('<span class="message-role assistant-role">Veris</span>',unsafe_allow_html=True)
        try:
            if previous and previous['question']==question and previous.get('citation_version')==3:
                result=dict(answer=previous['answer'],sources=[s['id'] for s in previous['sources']],excerpts={s['id']:s['excerpts'] for s in previous['sources']},status='answered'); evidence=previous['sources']
            elif key in st.session_state.cache: result=st.session_state.cache[key]
            else:
                if evidence: st.session_state.calls+=1
                result=respond(question,evidence,history)
                st.session_state.cache[key]=result
                if len(st.session_state.cache)>50: del st.session_state.cache[next(iter(st.session_state.cache))]
            sources=[dict(id=h['id'],filename=h['filename'],page=h['page'],excerpts=result.get('excerpts',{}).get(h['id'],[])) for h in evidence if h['id'] in result['sources']]
            message=dict(question=question,answer=result['answer'],sources=sources,scope=sorted(selected),citation_version=3)
        except Exception as exc:
            # Log only diagnostic metadata, never prompts, files or credentials.
            status=getattr(exc,'status_code',None)
            logging.getLogger('veris').warning('Answer failed: type=%s status=%s',type(exc).__name__,status)
            if isinstance(exc,ValueError):
                error_text='I could not produce a complete answer with valid source references. Please try again.'
            elif status==401:
                error_text='The API key was rejected. Check the connection in your app secrets.'
            elif status==429:
                error_text='The answer service reached a usage or rate limit. Please check the provider quota or try later.'
            elif 'Timeout' in type(exc).__name__:
                error_text='The answer service took too long. Please try again.'
            else:
                error_text='The answer service could not complete the request. Please try again. The server logs contain the error type.'
            message=dict(question=question,answer=error_text,sources=[],error=True,scope=sorted(selected))
    st.session_state.messages.append(message)
    st.session_state.messages=st.session_state.messages[-30:]
    st.rerun()
