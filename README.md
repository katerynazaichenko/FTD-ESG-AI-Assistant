### FTD-ESG-AI-Assistant
![Python](https://img.shields.io/badge/python-3.9+-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

This repository contains the project for an advanced AI assistant capable of performing accurate question-answering on dense Environmental, Social, and Governance (ESG) reports. The solution leverages a custom-built Retrieval-Augmented Generation (RAG) pipeline to deliver precise, context-aware answers without hallucination.

### Project Overview
The primary goal of this project is to tackle the challenge of information accessibility in lengthy and complex corporate ESG reports. By transforming unstructured PDF documents into a queryable knowledge base, this tool enables stakeholders to quickly find specific, reliable information without manually reading through hundreds of pages.

The project implements a full RAG workflow:

*   **Document Ingestion & Parsing**: Utilizes `MarkItDown` to efficiently extract and structure content from ESG reports in PDF format.
*   **Vectorization & Indexing**: The extracted content is segmented into manageable chunks, which are then converted into vector embeddings using OpenAI's models and stored in a searchable index.
*   **Advanced Retrieval & Reranking**: When a user asks a question, a hybrid retrieval system identifies relevant document chunks. These chunks are then reranked using a `ms-marco-MiniLM-L-6-v2` cross-encoder to prioritize the most contextually accurate passages.
*   **Generative Question Answering**: The top-ranked chunks are fed into a large language model (LLM) via a custom-engineered prompt, which instructs the model to generate an answer based *only* on the provided information.
*   **Automated Evaluation**: The entire pipeline's performance is rigorously assessed using the **RAGAS** framework, which measures key metrics like faithfulness, answer relevancy, and context precision.

### Key Results & Insights
The RAG pipeline was evaluated to ensure the generated answers are both accurate and grounded in the source document. The results highlight the system's strong performance in retrieving relevant context and generating faithful answers.

*   **High Faithfulness and Relevancy**: The model scores well on **Faithfulness (0.70)** and **Answer Relevancy (0.71)**, demonstrating its ability to generate answers that are both factually consistent with the source text and directly address the user's query.
*   **Excellent Context Precision**: With a **Context Precision of 0.80**, the retrieval system is highly effective at sourcing the correct information, which is critical for minimizing irrelevant noise and preventing hallucinations.
*   **Area for Improvement**: The **Context Recall score of 0.49** indicates an opportunity for future enhancement. While the retrieved context is highly relevant, the system could be improved to ensure all relevant information from the document is captured for more comprehensive answers.

| Metric              | Score |
| ------------------- | :---: |
| Faithfulness        | 0.70  |
| Answer Relevancy    | 0.71  |
| Context Precision   | 0.80  |
| Context Recall      | 0.49  |

### Repository Structure
```bash
.  
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── data
│   └── raw
│       └── esg_reports.csv
├── reports
│   └── figures
└── src
    ├── evaluate_rag_pipeline.py
    ├── llamaindex_rag_demo_2.py
    └── process_esg_dataset.py
```
### How to Run This Project
To set up and run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/katerynazaichenko/FTD-ESG-AI-Assistant.git
    cd FTD-ESG-AI-Assistant
    ```

2.  **Set up a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the scripts:**
    Execute the scripts from the `src/` directory. You will need to set up your OpenAI API key as an environment variable.

    *   First, process the ESG data to create the knowledge base:
        ```bash
        python src/process_esg_dataset.py
        ```
    *   Next, run the evaluation pipeline to test the model's performance:
        ```bash
        python src/evaluate_rag_pipeline.py
        ```
    *   Finally, launch the interactive Streamlit demo to ask your own questions:
        ```bash
        streamlit run src/llamadex_rag_demo_2.py
        ```

### Authors
*   Jessie Cameron
*   Gabriela Moravcikova
*   Kateryna Zaichenko
