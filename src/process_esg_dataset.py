import csv
import os
import logging
import yaml
from tqdm import tqdm
from pathlib import Path
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from llama_index.core.settings import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.prompts import PromptTemplate

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "openai_config_template.yaml")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def load_and_enhance_documents():
    logger.info("🔹 Loading documents with MarkItDown and vision support...")
    from markitdown import MarkItDown
    from openai import OpenAI as OpenAIClient
    from llama_index.core.schema import Document

    config = load_config()
    client = OpenAIClient(api_key=config['openai_api_key'])
    md = MarkItDown(llm_client=client, llm_model=config['completion_model'])

    folder_path = os.path.join(os.path.dirname(__file__), "data")
    docs = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            try:
                result = md.convert(file_path)
                text = getattr(result, 'structured_text', None) or getattr(result, 'text_content', '')
                if not text.strip():
                    raise ValueError("No text extracted from document.")
                docs.append(Document(text=text, metadata={"title": result.title or filename}))
            except Exception as e:
                logger.warning(f"⚠️ Failed to process {filename} with MarkItDown: {e}")

    logger.info(f"🔸 Loaded {len(docs)} documents from folder")
    return docs


def setup_rag_system():
    config = load_config()
    
    # ✅ Fix for LlamaIndex
    import openai
    openai.api_key = config['openai_api_key']

    # LlamaIndex model setup
    Settings.llm = OpenAI(model=config['completion_model'], temperature=0)
    Settings.embed_model = OpenAIEmbedding(model=config['embedding_model'])

    docs = load_and_enhance_documents()
    if not docs:
        logger.error("❌ No valid documents loaded. Aborting.")
        return None

    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(docs, include_metadata=True)
    logger.info(f"🔸 Created {len(nodes)} chunks")

    from llama_index.core import StorageContext

    persist_dir = os.path.join(os.path.dirname(__file__), "storage")

    # Create a fresh context without loading any existing files
    storage_context = StorageContext.from_defaults()

    # Build and persist index
    index = VectorStoreIndex(nodes, storage_context=storage_context)
    storage_context.persist(persist_dir=persist_dir)

    # Create retriever
    retriever = index.as_retriever(similarity_top_k=7)  # Slightly larger pool for reranker

    # Add re-ranker
    reranker_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker = SentenceTransformerRerank(top_n=3, model=reranker_model_name)

    custom_prompt_template = PromptTemplate(
    """You are a meticulous ESG Data Analyst Assistant. Your primary goal is to answer questions accurately based **solely and exclusively** on the provided context. Do not use any external knowledge or make assumptions beyond what is explicitly stated in the context.

    Instructions:
    1.  **Identify Key Information:** Carefully read the Question and then scan the Context to find the precise sentence(s) or data point(s) that directly answer the Question.
    2.  **Direct Answer Only:** Formulate your answer using only the information extracted from the Context.
        *   If the Question asks for a specific fact, number, or statement, your answer MUST be directly supported by a specific part of the Context.
    3.  **Handle Missing Information:** If, after careful review, the specific information to answer the Question is **not explicitly present** in the Context, you MUST respond with: "The provided context does not contain the specific information to answer this question." Do not try to infer or guess.
    4.  **Quoting (Optional but Preferred):** When providing an answer, if possible, quote the most relevant short phrase or sentence from the Context that directly supports your answer. For example: "Yes, according to the context, 'Nestlé remains committed to net zero by 2050.'"
    5.  **Formatting:** If the answer involves multiple points extracted from the context, use bullet points.
    6.  **Conciseness:** Be as concise as possible while being accurate and fully grounded in the provided context.

    Context:
    {context_str}

    Question:
    {query_str}

    Answer:
    """
    )

    # Create query engine with reranker
    query_engine = RetrieverQueryEngine.from_args(
    retriever=retriever,
    node_postprocessors=[reranker],
    text_qa_template=custom_prompt_template,
    )

    return query_engine


def process_dataset(input_csv_path, output_csv_path):
    query_engine = setup_rag_system()

    # Read input CSV
    with open(input_csv_path, 'r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        rows = list(reader)

    # Write results in RAGAS-compatible format
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as outfile:
        fieldnames = ['user_input', 'retrieved_contexts', 'response', 'reference']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in tqdm(rows, desc="Processing questions"):
            question = row['question']
            ground_truth = row['ground_truth_answer']
            response = query_engine.query(question)

            retrieved_contexts = [
                f"[{node.node.metadata.get('title', 'Unknown')}]: {node.node.text.strip()}"
                for node in response.source_nodes
            ]


            writer.writerow({
                'user_input': question,
                'retrieved_contexts': str(retrieved_contexts),
                'response': response.response,
                'reference': ground_truth
            })


if __name__ == "__main__":
    input_csv_path = os.path.join(os.path.dirname(__file__), "datasets", "curated_esg_dataset_totalenergies_v2.csv")
    output_csv_path = os.path.join(os.path.dirname(__file__), "datasets", "rag_evaluation_dataset.csv")

    logger.info("🚀 Starting dataset processing...")
    process_dataset(input_csv_path, output_csv_path)
    logger.info(f"✅ Processing complete. Results saved to {output_csv_path}")
