import streamlit as st
import os
import tempfile
import time

from langfuse import Langfuse

from src.rag import engine as shared_engine
from src.config import LLM_MODEL

st.set_page_config(
    page_title="RAG Chatbot", 
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    .source-box {
        font-size: 0.85em;
        padding: 8px;
        margin-top: 2px;
        border-radius: 5px;
        border-left: 3px solid #ff4b4b;
        background-color: rgba(128, 128, 128, 0.05);
    }
    div[data-testid="stToast"] {
        width: fit-content;
        min-width: 200px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Initialize Langfuse ---
if "langfuse" not in st.session_state:
    st.session_state.langfuse = Langfuse()

@st.cache_resource(show_spinner=False)
def load_rag_engine():
    return shared_engine

# --- Boot sequence ---
if "rag_engine" not in st.session_state:
    loader_placeholder = st.empty()
    with loader_placeholder.container():
        st.markdown("<div style='height: 20vh'></div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            with st.status("🚀 Booting up AI System...", expanded=True) as status:
                st.write("🔌 Connecting to LLM API...")
                time.sleep(0.5)
                st.write("📡 Initializing Observability (Langfuse v3)...")
                st.write("💾 Connecting to Vector Database...")
                st.session_state.rag_engine = load_rag_engine()
                status.update(label="✅ System Online!", state="complete", expanded=False)
                time.sleep(1)
    loader_placeholder.empty()

engine = st.session_state.rag_engine

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar ---
with st.sidebar:
    st.title("📄 Document AI")
    st.caption(f"Engine: **{LLM_MODEL}**")
    st.markdown("---")
    
    mode = st.radio("🤖 Assistant Mode", ["PDF Assistant", "Custom Prompt"])
    
    custom_prompt_input = None
    if mode == "Custom Prompt":
        st.info("Define the persona. The system will auto-inject context.")
        custom_prompt_input = st.text_area(
            "System Instruction:", 
            value="You are a sarcastic pirate. Answer questions based on the document.",
            height=100
        )
    
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])
    
    if uploaded_file:
        if st.button("⚡ Process & Index", type="primary", use_container_width=True):
            with st.status("Ingesting Document...", expanded=True) as status:
                st.write("📥 Reading file...")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                st.write("🧠 Vectorizing & Indexing...")
                try:
                    num_chunks = engine.ingest_file(tmp_path)
                    os.remove(tmp_path)
                    status.update(
                        label=f"✅ Indexed {num_chunks} chunks.", 
                        state="complete", 
                        expanded=False
                    )
                except Exception as e:
                    status.update(label="❌ Failed", state="error")
                    st.error(f"Error: {e}")

    st.markdown("---")
    
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    if st.button("🔥 Reset Database", type="secondary", use_container_width=True):
        if engine.clear_database():
            st.toast("Database cleared!", icon="🗑️")
            st.session_state.messages = []
            time.sleep(1)
            st.rerun()
        else:
            st.error("Failed to clear database.")

# --- Chat Interface ---
st.subheader("💬 Chat with your Knowledge Base")

for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.write(msg["content"])
        else:
            st.code(msg["content"], language=None, wrap_lines=True)
            col_sources, col_spacer, col_feedback = st.columns([6, 1, 2])
            
            with col_sources:
                if "sources" in msg and msg["sources"]:
                    with st.expander("📚 Sources / Context"):
                        for s in msg["sources"]:
                            page = s['meta'].get('page', 0) + 1
                            text_preview = s['text'].replace("\n", " ").strip()[:200]
                            st.markdown(
                                f"<div class='source-box'><b>Page {page}</b>: {text_preview}...</div>", 
                                unsafe_allow_html=True
                            )
            
            with col_feedback:
                key = f"fb_{idx}"
                score = st.feedback("thumbs", key=key)
                
                if score is not None:
                    trace_id = msg.get("trace_id")
                    if trace_id and trace_id != "unknown":
                        value = 1.0 if score == 1 else 0.0
                        
                        # === FIXED FEEDBACK SUBMISSION ===
                        feedback_sent = False
                        error_msg = None
                        
                        # Method 1: Try score()
                        try:
                            st.session_state.langfuse.score(
                                trace_id=trace_id,
                                name="user-feedback",
                                value=value,
                                comment="Streamlit UI"
                            )
                            st.session_state.langfuse.flush()
                            feedback_sent = True
                        except AttributeError:
                            pass
                        except Exception as e:
                            error_msg = str(e)
                        
                        # Method 2: Try create_score()
                        if not feedback_sent:
                            try:
                                st.session_state.langfuse.create_score(
                                    trace_id=trace_id,
                                    name="user-feedback",
                                    value=value,
                                    comment="Streamlit UI"
                                )
                                st.session_state.langfuse.flush()
                                feedback_sent = True
                            except Exception as e:
                                error_msg = str(e)
                        
                        # Method 3: Try REST API
                        if not feedback_sent:
                            try:
                                import httpx
                                
                                langfuse_host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
                                langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
                                langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY")
                                
                                if langfuse_public_key and langfuse_secret_key:
                                    url = f"{langfuse_host}/api/public/scores"
                                    auth = (langfuse_public_key, langfuse_secret_key)
                                    payload = {
                                        "traceId": trace_id,
                                        "name": "user-feedback",
                                        "value": value,
                                        "comment": "Streamlit UI"
                                    }
                                    
                                    with httpx.Client() as client:
                                        response = client.post(url, json=payload, auth=auth, timeout=10.0)
                                        response.raise_for_status()
                                    
                                    feedback_sent = True
                            except Exception as e:
                                error_msg = str(e)
                        
                        # Show result
                        if feedback_sent:
                            st.toast("Thanks for feedback!", icon="✨")
                        else:
                            st.warning(f"Failed to send feedback: {error_msg}")
                    else:
                        st.warning("No trace_id available for feedback")

# --- Input Handling ---
if prompt := st.chat_input("Ask about your PDF..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                history = st.session_state.messages[:-1]
                system_prompt_to_use = custom_prompt_input if mode == "Custom Prompt" else None
                
                response_text, sources, trace_id = engine.get_answer_with_sources(
                    query=prompt,
                    chat_history=history,
                    custom_system_prompt=system_prompt_to_use
                )
                
                # Typewriter effect
                placeholder = st.empty()
                full_res = ""
                for chunk in response_text.split():
                    full_res += chunk + " "
                    time.sleep(0.02)
                    placeholder.markdown(full_res + "▌")
                
                placeholder.code(full_res, language=None, wrap_lines=True)
                
                if sources:
                    with st.expander("📚 Sources / Context"):
                        for s in sources:
                            page = s['meta'].get('page', 0) + 1
                            text_preview = s['text'].replace("\n", " ").strip()[:200]
                            st.markdown(
                                f"<div class='source-box'><b>Page {page}</b>: {text_preview}...</div>", 
                                unsafe_allow_html=True
                            )

                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_res.strip(),
                    "sources": sources,
                    "trace_id": trace_id
                })
                st.rerun()

            except Exception as e:
                st.error(f"Error: {e}")