import os
import yaml

# Load OpenAI API key from YAML config before importing LlamaIndex modules
with open("../config/openai_config_template.yaml", "r") as f:
    config = yaml.safe_load(f)

os.environ["OPENAI_API_KEY"] = config["openai_api_key"]

import streamlit as st
from llama_index.core import StorageContext, load_index_from_storage
from dotenv import load_dotenv

load_dotenv()

# Clean text formatting

def clean_chunk_text(text):
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue  # skip blank lines
        if cleaned_lines and not cleaned_lines[-1].endswith(('.', ':', '?', '!', '”')):
            cleaned_lines[-1] += " " + stripped
        else:
            cleaned_lines.append(stripped)
    return "\n\n".join(cleaned_lines)

# Configure Streamlit layout and title
st.set_page_config(page_title="ESG AI Assistant", layout="wide")
st.title("🌍 ESG AI Assistant")
st.markdown(
    "<p style='font-size: 18px;'>Ask any question about the uploaded ESG report. "
    "Responses are strictly grounded in the report content using "
    "<strong>Retrieval-Augmented Generation (RAG)</strong>.</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown("Model: `gpt-4o-mini`")
    st.markdown("Embedding: `text-embedding-3-small`")

# Load pre-built index
persist_dir = os.path.join(os.path.dirname(__file__), "storage")
if not os.path.exists(persist_dir):
    st.error("❌ No index found. Please run indexing first.")
else:
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context)
    query_engine = index.as_query_engine(similarity_top_k=3)

    # Custom-styled question prompt
    st.markdown(
        "<h4 style='margin-bottom: 0.2em; font-size: 20px; font-weight: 600;'>📝 Ask a question about the ESG report:</h4>",
        unsafe_allow_html=True,
    )
    question = st.text_area("", height=100, placeholder="e.g. What is the company's main sustainability focus?")

    if question:
        with st.spinner("🤖 Generating answer..."):
            response = query_engine.query(question)

            # Display the answer
            st.markdown("### 🧠 Answer")
            st.success(response.response)

            # Show retrieved context
            # Show the retrieved context chunks
            # Show retrieved context chunks
            st.markdown("### 📚 Retrieved Contexts")
            for i, node in enumerate(response.source_nodes):
                with st.expander(f"Chunk {i+1}", expanded=False):
                    cleaned_text = clean_chunk_text(node.node.text)
                    st.markdown(
                        f"""
                        <div style="white-space: pre-wrap;
                                    line-height: 1.6;
                                    font-size: 0.95rem;
                                    padding: 1rem;
                                    max-width: 100%;
                                    overflow-x: auto;">
                            {cleaned_text}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


# Footer
st.markdown("---")
st.caption("🔍 Built by Jessie Cameron, Gabriela Moravcikova & Kateryna Zaichenko for the FTD ESG project | Powered by LlamaIndex & OpenAI")