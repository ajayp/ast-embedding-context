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
  
## Proof of Concept Notes

* The "vector database" is just an in-memory list. A real application would use a dedicated vector database (e.g., Chroma, Pinecone, Weaviate, Qdrant) for scalability and efficient searching.
* The AST feature extraction is very basic. More sophisticated analysis could identify control flow, variable usage, dependencies, etc.
* The use of AST features in retrieval is currently just for highlighting in the output. In a real system, AST features could be used for re-ranking search results, filtering, or providing structured input to a Large Language Model (LLM) for deeper analysis.
* The sample data is small and manually created. A real system would ingest code from a repository and potentially stack trace data from monitoring systems.
* The similarity threshold (`SIMILARITY_THRESHOLD`) is a tunable parameter that affects retrieval strictness.

## Setup

1.  **Clone the repository (or save the code):** Save the provided Python code as a `.py` file (e.g., `error_context_retrieval.py`).
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

## Example Output

Running the script will produce output similar to the following. Note that the exact similarity scores might vary slightly depending on the library versions and underlying hardware, but the retrieved items and their relative scores should be consistent.

```text
Parsing code snippets and extracting AST features...
Parsed AST and extracted features for user_service.py::process_user_input: {'has_try_except': False, 'has_if': True, 'has_for': False, 'has_while': False, 'function_names_called': ['isinstance', 'get']}
Parsed AST and extracted features for user_service.py::save_user_to_db: {'has_try_except': False, 'has_if': False, 'has_for': False, 'has_while': False, 'function_names_called': ['connect_db', 'insert', 'close']}
Parsed AST and extracted features for database_utils.py::connect_db: {'has_try_except': False, 'has_if': True, 'has_for': False, 'has_while': False, 'function_names_called': ['print', 'random']}
Parsed AST and extracted features for api_handler.py::handle_request: {'has_try_except': True, 'has_if': False, 'has_for': False, 'has_while': False, 'function_names_called': ['process_user_input', 'save_user_to_db', 'print', 'str']}
Parsed AST and extracted features for config_loader.py::validate_config: {'has_try_except': False, 'has_if': True, 'has_for': False, 'has_while': False, 'function_names_called': ['get']}
Parsed AST and extracted features for order_service.py::process_order_data: {'has_try_except': False, 'has_if': True, 'has_for': False, 'has_while': False, 'function_names_called': ['isinstance', 'get']}

Populating vector database with code and stack frame embeddings (including AST features for code)...
Vector database populated with 13 items.

--- Running POC with Error Trace 1 (Database Connection Error) ---
Simulated Error Message: Database connection failed

Searching context for Frame 1: File "api_handler.py", line 11, in handle_request...
Found relevant context:
  - Code Snippet (Score: 0.72XX): api_handler.py::handle_request (Structurally Relevant: Has Try/Except)
  - Code Snippet (Score: 0.6XXX): user_service.py::save_user_to_db
  - Similar Stack Frame (Score: 0.85XX): File "api_handler.py", line 11, in handle_request
  - Similar Stack Frame (Score: 0.7XXX): File "user_service.py", line 14, in save_user_to_db
  - Similar Stack Frame (Score: 0.65XX): File "database_utils.py", line 6, in connect_db

Searching context for Frame 2: File "user_service.py", line 14, in save_user_to_db...
Found relevant context:
  - Code Snippet (Score: 0.7XXX): user_service.py::save_user_to_db
  - Code Snippet (Score: 0.6XXX): api_handler.py::handle_request (Structurally Relevant: Has Try/Except)
  - Similar Stack Frame (Score: 0.89XX): File "user_service.py", line 14, in save_user_to_db
  - Similar Stack Frame (Score: 0.7XXX): File "api_handler.py", line 11, in handle_request
  - Similar Stack Frame (Score: 0.65XX): File "database_utils.py", line 6, in connect_db

Searching context for Frame 3: File "database_utils.py", line 6, in connect_db...
Found relevant context:
  - Code Snippet (Score: 0.8XXX): database_utils.py::connect_db (Structurally Relevant: Has Conditional)
  - Similar Stack Frame (Score: 0.9XXX): File "database_utils.py", line 6, in connect_db
  - Similar Stack Frame (Score: 0.65XX): File "user_service.py", line 14, in save_user_to_db

--- Retrieved Context for Error Trace 1 ---
Relevant Code Snippets:
- api_handler.py::handle_request (Score: 0.72XX) (Structurally Relevant: Has Try/Except)
- user_service.py::save_user_to_db (Score: 0.7XXX)
- database_utils.py::connect_db (Score: 0.8XXX) (Structurally Relevant: Has Conditional)

Semantically Similar Stack Frames:
- File "api_handler.py", line 11, in handle_request (Score: 0.85XX)
- File "user_service.py", line 14, in save_user_to_db (Score: 0.89XX)
- File "database_utils.py", line 6, in connect_db (Score: 0.9XXX)


--- Running POC with Error Trace 2 (User Input Validation Error) ---
Simulated Error Message: Input must be a dictionary

Searching context for Frame 1: File "api_handler.py", line 10, in handle_request...
Found relevant context:
  - Code Snippet (Score: 0.7XXX): api_handler.py::handle_request (Structurally Relevant: Has Try/Except)
  - Code Snippet (Score: 0.7XXX): user_service.py::process_user_input (Structurally Relevant: Has Conditional)
  - Similar Stack Frame (Score: 0.8XXX): File "api_handler.py", line 10, in handle_request
  - Similar Stack Frame (Score: 0.6XXX): File "user_service.py", line 5, in process_user_input
  - Similar Stack Frame (Score: 0.55XX): File "user_service.py", line 9, in process_user_input

Searching context for Frame 2: File "user_service.py", line 5, in process_user_input...
Found relevant context:
  - Code Snippet (Score: 0.8XXX): user_service.py::process_user_input (Structurally Relevant: Has Conditional)
  - Code Snippet (Score: 0.6XXX): order_service.py::process_order_data (Structurally Relevant: Has Conditional)
  - Similar Stack Frame (Score: 0.9XXX): File "user_service.py", line 5, in process_user_input
  - Similar Stack Frame (Score: 0.7XXX): File "order_service.py", line 8, in process_order_data
  - Similar Stack Frame (Score: 0.6XXX): File "user_service.py", line 9, in process_user_input

--- Retrieved Context for Error Trace 2 ---
Relevant Code Snippets:
- api_handler.py::handle_request (Score: 0.7XXX) (Structurally Relevant: Has Try/Except)
- user_service.py::process_user_input (Score: 0.8XXX) (Structurally Relevant: Has Conditional)
- order_service.py::process_order_data (Score: 0.6XXX) (Structurally Relevant: Has Conditional)

Semantically Similar Stack Frames:
- File "api_handler.py", line 10, in handle_request (Score: 0.8XXX)
- File "user_service.py", line 5, in process_user_input (Score: 0.9XXX)
- File "user_service.py", line 9, in process_user_input (Score: 0.55XX)
- File "order_service.py", line 8, in process_order_data (Score: 0.7XXX)


--- Running POC with Error Trace 3 (Order Input Validation Error - Semantic Similarity Demo) ---
Simulated Error Message: Order data must be a dictionary

Searching context for Frame 1: File "some_api.py", line 25, in handle_order_request...
Found relevant context:
  - Code Snippet (Score: 0.5XXX): api_handler.py::handle_request (Structurally Relevant: Has Try/Except)
  - Similar Stack Frame (Score: 0.5XXX): File "api_handler.py", line 10, in handle_request
No relevant context found for this frame above the threshold. # This might appear depending on scores below threshold

Searching context for Frame 2: File "order_service.py", line 8, in process_order_data...
Found relevant context:
  - Code Snippet (Score: 0.8XXX): order_service.py::process_order_data (Structurally Relevant: Has Conditional)
  - Code Snippet (Score: 0.7XXX): user_service.py::process_user_input (Structurally Relevant: Has Conditional)
  - Similar Stack Frame (Score: 0.9XXX): File "order_service.py", line 8, in process_order_data
  - Similar Stack Frame (Score: 0.7XXX): File "user_service.py", line 5, in process_user_input

--- Retrieved Context for Error Trace 3 ---
Relevant Code Snippets:
- order_service.py::process_order_data (Score: 0.8XXX) (Structurally Relevant: Has Conditional)
- user_service.py::process_user_input (Score: 0.7XXX) (Structurally Relevant: Has Conditional)
- api_handler.py::handle_request (Score: 0.5XXX) (Structurally Relevant: Has Try/Except) # Might be retrieved depending on threshold

Semantically Similar Stack Frames:
- File "order_service.py", line 8, in process_order_data (Score: 0.9XXX)
- File "user_service.py", line 5, in process_user_input (Score: 0.7XXX)
- File "api_handler.py", line 10, in handle_request (Score: 0.5XXX) # Might be retrieved depending on threshold
