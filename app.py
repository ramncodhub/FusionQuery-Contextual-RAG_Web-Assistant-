import os
import tempfile
import streamlit as st
from ingest import clear_vectorstore, process_and_store_document
from llm_operations import run_chat_pipeline
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="RAG & Web AI Assistant", page_icon="⚡", layout="wide"
)

st.title("⚡ AI Assistant with RAG & Self-Refinement")
st.caption(
    "Cross-Encoder Reranking | Gated Web Fallback | Automated Hallucination Critique"
)

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_evaluation_expander(context_type: str, eval_data: dict):
    """Renders evaluation metrics and self-refinement alerts."""
    with st.expander("📊 AI Evaluation & Retrieval Diagnostics"):
        st.caption(f"**Source Used:** `{context_type or 'N/A'}`")
        
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Correctness", f"{eval_data.get('correctness_score', 'N/A')}/100"
        )
        c2.metric("Relevance", f"{eval_data.get('relevance_score', 'N/A')}/100")
        c3.metric(
            "Hallucination Check", eval_data.get("hallucination_check", "N/A")
        )
        
        if eval_data.get("was_refined", False):
            st.warning("🔄 **Self-Refinement Triggered:** The draft contained inaccuracies or hallucinations. The model critiqued and corrected the answer.")
            if "original_critique" in eval_data:
                st.info(f"**Critique:** {eval_data['original_critique']}")
        else:
            st.success("✅ **Strictly Grounded:** The output passed factual entailment checks.")
            
        st.caption(f"**Remarks:** {eval_data.get('explanation', 'N/A')}")


# Sidebar
with st.sidebar:
    st.header("📄 Document Management")
    uploaded_file = st.file_uploader("Upload a PDF for QA", type=["pdf"])

    if uploaded_file is not None and st.button("Process Document", type="primary"):
        with st.spinner("Indexing PDF into FAISS with dense chunking..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            try:
                process_and_store_document(tmp_path)
                st.success("PDF indexed with cross-encoder compatibility!")
            except Exception as e:
                st.error(f"Error indexing PDF: {e}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    st.divider()

    if st.button("🗑️ Remove Document & Clear Vector DB"):
        clear_vectorstore()
        st.success("Vector index cleared!")
        st.rerun()

    if st.button("🧹 Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# History Render
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "evaluation" in message:
            render_evaluation_expander(
                message.get("context_type"), message["evaluation"]
            )

# Input Loop
if prompt := st.chat_input("Ask a question about your PDF or a general web topic..."):
    st.chat_message("user").markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Reranking context and verifying factual grounding..."):
            res = run_chat_pipeline(prompt, st.session_state.messages)

            st.markdown(res["answer"])

            eval_data = res.get("evaluation", {})
            render_evaluation_expander(res.get("context_type"), eval_data)

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": res["answer"],
            "context_type": res.get("context_type"),
            "evaluation": eval_data,
        }
    )
