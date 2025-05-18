import os
import pandas as pd
import yaml
import logging
import ast
from pathlib import Path
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from ragas import EvaluationDataset
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from ragas.llms import LlamaIndexLLMWrapper
from ragas.integrations.llama_index import evaluate

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_ragas_evaluation_dataset(csv_path):
    """
    Loads a CSV with columns: user_input, retrieved_contexts, response, reference
    and returns a RAGAS EvaluationDataset.
    """
def load_ragas_evaluation_dataset(csv_path):
    """
    Loads a CSV with columns: question, retrieved_contexts, answer, reference
    and returns a RAGAS EvaluationDataset.
    """
    df = pd.read_csv(csv_path)
    dataset = []
    for _, row in df.iterrows():
        contexts = row["retrieved_contexts"]
        if isinstance(contexts, str):
            try:
                parsed = ast.literal_eval(contexts)
                if isinstance(parsed, list):
                    contexts = parsed
                else:
                    contexts = [parsed]
            except Exception:
                contexts = [contexts]
        dataset.append({
            "user_input": row["user_input"],
            "retrieved_contexts": contexts,
            "response": row["response"],
            "reference": row["reference"],
        })
    return EvaluationDataset.from_list(dataset)

def load_config():
    """Load configuration from YAML file."""
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "openai_config_template.yaml")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    # Load configuration
    config = load_config()
    
    # Set OpenAI API key from config
    os.environ["OPENAI_API_KEY"] = config['openai_api_key']
    
    # Load documents
    logger.info("Loading documents...")
    documents = SimpleDirectoryReader("./data").load_data()
    logger.info(f"Loaded {len(documents)} documents")

    # Initialize LLM and embedding model using config
    generator_llm = OpenAI(model=config['completion_model'])
    embeddings = OpenAIEmbedding(model=config['embedding_model'])

    # Build VectorStoreIndex and Query Engine
    logger.info("Building vector store index...")
    vector_index = VectorStoreIndex.from_documents(documents)
    query_engine = vector_index.as_query_engine()

    # Define evaluation metrics
    logger.info("Setting up evaluation metrics...")
    evaluator_llm = LlamaIndexLLMWrapper(OpenAI(model=config['completion_model']))
    metrics = [
        Faithfulness(llm=evaluator_llm),
        AnswerRelevancy(llm=evaluator_llm),
        ContextPrecision(llm=evaluator_llm),
        ContextRecall(llm=evaluator_llm),
    ]

    csv_path = "./datasets/rag_evaluation_dataset.csv"
    ragas_dataset = load_ragas_evaluation_dataset(csv_path)

    # Evaluate the Query Engine
    logger.info("Starting evaluation...")
    result = evaluate(
        query_engine=query_engine,
        metrics=metrics,
        dataset=ragas_dataset,
    )

    # Display evaluation results
    logger.info("Evaluation complete. Results:")
    print(result)

    # Convert results to pandas DataFrame
    result_df = result.to_pandas()
    print(result_df.head())

    # Calculate and print mean scores for each metric
    mean_scores = result_df.mean(numeric_only=True)
    print("\nMean scores for each metric:")
    print(mean_scores)

    # Save all results to CSV
    result_df.to_csv("ragas_evaluation_results.csv", index=False)
    print("All detailed results saved to ragas_evaluation_results.csv")

if __name__ == "__main__":
    main()
