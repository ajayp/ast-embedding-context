# Error Context Retrieval Proof of Concept

This POC provides a foundational idea for how semantic search and structural code analysis can be combined to build more intelligent error analysis and debugging tools.

This is a simple Python script demonstrating a proof of concept for retrieving relevant code context and similar historical execution steps (stack frames) based on a given error stack trace. 

It utilizes sentence embeddings for semantic similarity and Abstract Syntax Tree (AST) parsing for extracting structural features from code.

## Features

* **Sentence Embeddings:** Uses the `sentence-transformers` library to generate vector embeddings for code snippets and stack trace frames to capture semantic meaning.
* **AST Parsing:** Parses Python code snippets into Abstract Syntax Trees to extract structural features (e.g., presence of `try`/`except`, `if`, `for`/`while` loops, function calls).
* **Simulated Vector Database:** Stores code and stack frame data along with their embeddings and AST features in a simple in-memory list.
* **Context Retrieval:** Given a new error stack trace, it embeds its frames and searches the simulated vector database for similar code snippets and historical stack frames based on cosine similarity.
* **AST Feature Hinting:** Demonstrates how extracted AST features could potentially be used to highlight structural relevance (e.g., indicating if a retrieved code snippet contains error handling logic).

## How it Works

1.  **Data Preparation:** Sample code snippets and simulated historical stack trace frames are defined.
2.  **AST Analysis:** Each code snippet is parsed into an AST, and a predefined set of structural features is extracted.
3.  **Embedding:** Sentence embeddings are generated for the content of both code snippets and stack trace frames using a pre-trained model (`all-MiniLM-L6-v2`).
4.  **Storage:** The original content, metadata, embeddings, and AST features (for code) are stored in a simple in-memory list acting as a simulated vector database.
5.  **Retrieval:** When a new error stack trace occurs:
    * Each frame of the new stack trace is embedded.
    * These frame embeddings are used to query the simulated vector database.
    * Items (code or stack frames) in the database with embeddings similar (above a threshold) to the query frame embeddings are retrieved.
    * The results are presented, including the similarity score and simple indicators based on AST features for code snippets.

## Setup

1.  **Clone the repository (or save the code):** Save the provided Python code as a `.py` file (e.g., `error_context_poc.py`).
2.  **Install dependencies:** You need Python 3.6+ and the following libraries.

    ```bash
    pip install numpy sentence-transformers torch
    ```

## How to Run

1.  Navigate to the directory where you saved the script.
2.  Run the script directly from your terminal:

    ```bash
    python error_context_retrieval.py
    ```

The script will output the process of parsing, embedding, populating the simulated database, and then demonstrate context retrieval for three different simulated error traces.

## Proof of Concept Notes

* The "vector database" is just an in-memory list. A real application would use a dedicated vector database (e.g., Chroma, Pinecone, Weaviate, Qdrant) for scalability and efficient searching.
* The AST feature extraction is very basic. More sophisticated analysis could identify control flow, variable usage, dependencies, etc.
* The use of AST features in retrieval is currently just for highlighting in the output. In a real system, AST features could be used for re-ranking search results, filtering, or providing structured input to a Large Language Model (LLM) for deeper analysis.
* The sample data is small and manually created. A real system would ingest code from a repository and potentially stack trace data from monitoring systems.
* The similarity threshold (`SIMILARITY_THRESHOLD`) is a tunable parameter that affects retrieval strictness.

