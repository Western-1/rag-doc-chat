import streamlit as st
import os
import tempfile
import time

# --- MLOps Imports ---
from langfuse import Langfuse
from langfuse.callback import CallbackHandler

# --- OPTIMIZATION: Import shared engine ---
from src.rag import engine as shared_engine
from src.config import LLM_MODEL

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="RAG Chatbot", 
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Injection
st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    
    /* Source box styling */
    .source-box {
        font-size: 0.85em;
        padding: 8px;
        margin-top: 2px;
        border-radius: 5px;
        border-left: 3px solid #ff4b4b;
        background-color: rgba(128, 128, 128, 0.05);
    }
    
    /* Toast notification styling fix */
    div[data-testid="stToast"] {
        width: fit-content;
        min-width: 200px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. INIT MLOPS ---
if "langfuse" not in st.session_state:
    st.session_state.langfuse = Langfuse()

# --- 3. ENGINE LOADER ---
@st.cache_resource(show_spinner=False)
def load_rag_engine():
    # Return the globally instantiated engine to save memory
    return shared_engine

# --- 4. CUSTOM LOADING SCREEN ---
if "rag_engine" not in st.session_state:
    loader_placeholder = st.empty()
    with loader_placeholder.container():
        st.markdown("<div style='height: 20vh'></div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            with st.status("🚀 Booting up AI System...", expanded=True) as status:
                st.write("🔌 Connecting to 20B LLM...")
                time.sleep(0.5)
                st.write("📡 Initializing Observability (Langfuse)...")
                st.write("💾 Connecting to Qdrant Database...")
                st.session_state.rag_engine = load_rag_engine()
                status.update(label="✅ System Online!", state="complete", expanded=False)
                time.sleep(1)
    loader_placeholder.empty()

engine = st.session_state.rag_engine

# --- 5. SESSION HISTORY ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 6. SIDEBAR ---
with st.sidebar:
    st.title("📄 Document AI")
    st.caption(f"Engine: **{LLM_MODEL}**")
    
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])
    
    if uploaded_file:
        if st.button("⚡ Process & Index", type="primary", use_container_width=True):
            with st.status("Ingesting Document...", expanded=True) as status:
                st.write("📥 Reading file...")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                st.write("🧠 Vectorizing & Deduping...")
                try:
                    num_chunks = engine.ingest_file(tmp_path)
                    os.remove(tmp_path)
                    status.update(label=f"✅ Indexed {num_chunks} chunks.", state="complete", expanded=False)
                except Exception as e:
                    status.update(label="❌ Failed", state="error")
                    st.error(f"Error: {e}")

    st.markdown("---")
    
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.button("🔥 Reset Database (Delete All)", type="secondary", use_container_width=True):
        if engine.clear_database():
            st.toast("Database cleared!", icon="🗑️")
            st.session_state.messages = []
            time.sleep(1)
            st.rerun()
        else:
            st.error("Failed to clear database.")

# --- 7. CHAT INTERFACE ---
st.subheader("💬 Chat with your Knowledge Base")

for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.write(msg["content"])
        else:
            # 1. Response Text (st.code provides a native copy button)
            st.code(msg["content"], language=None, wrap_lines=True)
            
            # Layout: Sources (Left/Wide) | Spacer | Feedback (Right/Narrow)
            col_sources, col_spacer, col_feedback = st.columns([6, 1, 2])
            
            with col_sources:
                if "sources" in msg and msg["sources"]:
                    with st.expander("📚 Sources / Context"):
                        for s in msg["sources"]:
                            page = s['meta'].get('page', 0) + 1
                            text_preview = s['text'].replace("\n", " ").strip()[:200]
                            st.markdown(f"<div class='source-box'><b>Page {page}</b>: {text_preview}...</div>", unsafe_allow_html=True)

            with col_feedback:
                # 2. User Feedback (Thumbs Up/Down)
                key = f"fb_{idx}"
                score = st.feedback("thumbs", key=key)
                
                if score is not None:
                    trace_id = msg.get("trace_id")
                    if trace_id:
                        # Convert Streamlit score (0/1) to Langfuse format
                        value = 1.0 if score == 1 else 0.0
                        
                        st.session_state.langfuse.score(
                            trace_id=trace_id,
                            name="user-feedback",
                            value=value,
                            comment="Streamlit UI"
                        )
                        st.toast("Thanks for feedback!", icon="✨")

# --- 8. INPUT HANDLING ---
if prompt := st.chat_input("Ask about your PDF..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Initialize Langfuse Handler for this run
                langfuse_handler = CallbackHandler()
                
                response_text, sources, trace_id = engine.get_answer_with_sources(
                    prompt, 
                    callbacks=[langfuse_handler]
                )
                
                # Streaming/Typing effect simulation
                placeholder = st.empty()
                full_res = ""
                for chunk in response_text.split():
                    full_res += chunk + " "
                    time.sleep(0.02)
                    placeholder.markdown(full_res + "▌")
                
                # Final render using st.code to enable copy functionality
                placeholder.code(full_res, language=None, wrap_lines=True)
                
                # Optional: Display sources immediately after generation
                if sources:
                    with st.expander("📚 Sources / Context"):
                        for s in sources:
                            page = s['meta'].get('page', 0) + 1
                            text_preview = s['text'].replace("\n", " ").strip()[:200]
                            st.markdown(f"<div class='source-box'><b>Page {page}</b>: {text_preview}...</div>", unsafe_allow_html=True)

                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_res,
                    "sources": sources,
                    "trace_id": trace_id
                })
                
                # Rerun to ensure the feedback widget is rendered immediately
                st.rerun()

            except Exception as e:
                st.error(f"Error: {e}")