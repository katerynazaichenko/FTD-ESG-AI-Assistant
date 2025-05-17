# FTD-ESG-AI-Assistant

A functional AI assistant that answers questions about ESG reports using Retrieval-Augmented Generation (RAG).

## 🧠 What It Does

This assistant takes an ESG report in PDF format, extracts its content using MarkItDown, and enables accurate question-answering with a custom LLM-powered RAG pipeline. Answers are strictly based on the content of the document — no hallucinations.

## 🔧 How It Works

- **Document Parsing**: Extracts structured content from ESG PDF reports.
- **Vector Indexing**: Splits the content into chunks and embeds them using OpenAI embeddings.
- **Retrieval + Reranking**: Retrieves the most relevant chunks and reranks them using a cross-encoder (`ms-marco-MiniLM-L-6-v2`).
- **Question Answering**: Uses a custom prompt with OpenAI's LLM to answer based on top-ranked context.
- **Evaluation**: Outputs are evaluated using [RAGAS](https://github.com/explodinggradients/ragas) metrics (faithfulness, relevancy, precision, recall).

## 📊 Example Metrics

| Metric             | Score     |
|--------------------|-----------|
| Faithfulness        | 0.70      |
| Answer Relevancy    | 0.71      |
| Context Precision   | 0.80      |
| Context Recall      | 0.49      |

## 🚀 Getting Started

```bash
cd src
pip install -r requirements.txt
python process_esg_dataset.py
python evaluate_rag_pipeline.py
streamlit run llamaindex_rag_demo_2.py
