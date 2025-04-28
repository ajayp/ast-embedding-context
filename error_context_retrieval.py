import numpy as np
from sentence_transformers import SentenceTransformer, util
import random
import ast # Import the AST module
import inspect

# --- Configuration ---
# Using a pre-trained model for generating embeddings
# 'all-MiniLM-L6-v2' is a good balance of speed and performance for sentence embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Threshold for considering embeddings 'similar'
SIMILARITY_THRESHOLD = 0.5 # Cosine similarity threshold

# --- AST Parsing and Feature Extraction Helper ---

def extract_ast_features(tree):
    """Traverses AST and extracts relevant structural features."""
    features = {
        'has_try_except': False,
        'has_if': False,
        'has_for': False,
        'has_while': False,
        'function_names_called': [], # Simple list of function call names
        # Add more features as needed (e.g., variable assignments, imports)
    }

    if tree is None:
        return features # Return default features if parsing failed

    # Simple visitor pattern to find specific node types
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            features['has_try_except'] = True
        elif isinstance(node, ast.If):
            features['has_if'] = True
        elif isinstance(node, ast.For):
            features['has_for'] = True
        elif isinstance(node, ast.While):
            features['has_while'] = True
        elif isinstance(node, ast.Call):
             # Extract function name from a Call node
             if isinstance(node.func, ast.Name):
                 features['function_names_called'].append(node.func.id)
             elif isinstance(node.func, ast.Attribute):
                 # Handle method calls like obj.method()
                 features['function_names_called'].append(node.func.attr)


    # Remove duplicates from function_names_called
    features['function_names_called'] = list(set(features['function_names_called']))

    return features


def parse_code_and_extract_ast_features(code_string):
    """Parses code to AST and extracts features."""
    tree = None
    features = {}
    try:
        tree = ast.parse(code_string)
        features = extract_ast_features(tree)
    except SyntaxError as e:
        print(f"Warning: Could not parse code snippet into AST due to SyntaxError: {e}")
        # features will be empty/default if parsing fails
    return tree, features

# --- Data Simulation ---

# Sample Code Snippets (representing chunks from a repo)
# Now we'll add placeholders for AST object and extracted features
code_snippets = [
    {"id": "code_1", "content": "def process_user_input(data):\n    # Validate input data\n    if not isinstance(data, dict):\n        raise ValueError('Input must be a dictionary')\n    user = data.get('user')\n    if not user:\n        raise ValueError('User data is missing')\n    return user", "metadata": {"file": "user_service.py", "function": "process_user_input"}, "ast": None, "ast_features": {}},
    {"id": "code_2", "content": "def save_user_to_db(user_data):\n    # Connect to database\n    db_connection = connect_db()\n    # Insert user data\n    db_connection.insert('users', user_data)\n    db_connection.close()", "metadata": {"file": "user_service.py", "function": "save_user_to_db"}, "ast": None, "ast_features": {}},
    {"id": "code_3", "content": "def connect_db():\n    # Simulate a database connection\n    print('Connecting to database...')\n    # This function might sometimes fail due to network issues\n    if random.random() < 0.1: # Simulate 10% failure rate\n        raise ConnectionError('Database connection failed')\n    return {'status': 'connected'}", "metadata": {"file": "database_utils.py", "function": "connect_db"}, "ast": None, "ast_features": {}},
    {"id": "code_4", "content": "def handle_request(request_data):\n    try:\n        user_info = process_user_input(request_data)\n        save_user_to_db(user_info)\n        return {'status': 'success'}\n    except Exception as e:\n        print(f'Error handling request: {e}')\n        # In a real system, this would log the error and stack trace\n        return {'status': 'error', 'message': str(e)}", "metadata": {"file": "api_handler.py", "function": "handle_request"}, "ast": None, "ast_features": {}},
    {"id": "code_5", "content": "def validate_config(config):\n    if not config.get('database_url'):\n        raise ValueError('Database URL missing in config')\n    # More validation logic...", "metadata": {"file": "config_loader.py", "function": "validate_config"}, "ast": None, "ast_features": {}},
     {"id": "code_6", "content": "def process_order_data(data):\n    # Similar validation logic for order data\n    if not isinstance(data, dict):\n        raise TypeError('Order data must be a dictionary')\n    order_id = data.get('order_id')\n    if not order_id:\n        raise ValueError('Order ID is missing')\n    return order_id", "metadata": {"file": "order_service.py", "function": "process_order_data"}, "ast": None, "ast_features": {}}, # Semantically similar to code_1 (validation)
]

# --- Parse AST and Extract Features for Code Snippets ---
print("Parsing code snippets and extracting AST features...")
for code_item in code_snippets:
    tree, features = parse_code_and_extract_ast_features(code_item['content'])
    code_item['ast'] = tree # Store the tree (optional in real DB)
    code_item['ast_features'] = features # Store the extracted features

    if tree:
        print(f"Parsed AST and extracted features for {code_item['metadata']['file']}::{code_item['metadata']['function']}: {features}")


# Simulated Stack Trace Frames (representing steps in execution)
# These remain the same as they represent runtime information, not static code structure
simulated_stack_frames = [
    {"id": "st_frame_1", "content": "File \"api_handler.py\", line 10, in handle_request\n    user_info = process_user_input(request_data)", "metadata": {"file": "api_handler.py", "line": 10, "function": "handle_request"}},
    {"id": "st_frame_2", "content": "File \"user_service.py\", line 5, in process_user_input\n    if not isinstance(data, dict):", "metadata": {"file": "user_service.py", "line": 5, "function": "process_user_input"}},
    {"id": "st_frame_3", "content": "File \"user_service.py\", line 9, in process_user_input\n    if not user:", "metadata": {"file": "user_service.py", "line": 9, "function": "process_user_input"}},
    {"id": "st_frame_4", "content": "File \"api_handler.py\", line 11, in handle_request\n    save_user_to_db(user_info)", "metadata": {"file": "api_handler.py", "line": 11, "function": "handle_request"}},
    {"id": "st_frame_5", "content": "File \"user_service.py\", line 14, in save_user_to_db\n    db_connection = connect_db()", "metadata": {"file": "user_service.py", "line": 14, "function": "save_user_to_db"}},
    {"id": "st_frame_6", "content": "File \"database_utils.py\", line 6, in connect_db\n    raise ConnectionError('Database connection failed')", "metadata": {"file": "database_utils.py", "line": 6, "function": "connect_db"}},
     # Another stack trace frame, potentially from a different error or service, but semantically similar step
    {"id": "st_frame_7", "content": "File \"order_service.py\", line 8, in process_order_data\n    if not isinstance(data, dict):", "metadata": {"file": "order_service.py", "line": 8, "function": "process_order_data"}}, # Semantically similar to st_frame_2 (input validation check)
]


# --- Embedding and Storage (Simulated Vector DB) ---

# In-memory storage for embeddings and original data
vector_db = []

def add_to_vector_db(data_item, item_type):
    """Generates embedding and adds item to the simulated vector DB."""
    content = data_item['content']
    embedding = embedding_model.encode(content, convert_to_tensor=True)

    db_entry = {
        "id": data_item["id"],
        "type": item_type, # 'code' or 'stack_frame'
        "content": content,
        "metadata": data_item["metadata"],
        "embedding": embedding
    }

    # Add AST features to the entry if it's a code item
    if item_type == 'code' and 'ast_features' in data_item:
         db_entry['ast_features'] = data_item['ast_features']


    vector_db.append(db_entry)

# Populate the vector database
print("\nPopulating vector database with code and stack frame embeddings (including AST features for code)...")
for code in code_snippets:
    add_to_vector_db(code, 'code')

for frame in simulated_stack_frames:
    add_to_vector_db(frame, 'stack_frame')

print(f"Vector database populated with {len(vector_db)} items.")

# --- Retrieval Logic ---

def retrieve_context(error_stack_trace, error_message, k=5):
    """
    Simulates retrieving relevant context from the vector DB based on an error stack trace.
    Retrieves relevant code chunks and semantically similar stack frames.
    Demonstrates using AST features to highlight relevant code.
    """
    print("\n--- Retrieving Context for New Error ---")
    print(f"Simulated Error Message: {error_message}") # Print the error message for context

    # 1. Embed the frames of the new error stack trace
    error_frame_embeddings = [embedding_model.encode(frame_content, convert_to_tensor=True) for frame_content in error_stack_trace]

    retrieved_code = {}
    retrieved_stack_frames = {}

    # 2. For each frame in the error stack trace, find relevant context
    for i, frame_embedding in enumerate(error_frame_embeddings):
        print(f"\nSearching context for Frame {i+1}: {error_stack_trace[i].splitlines()[0]}...") # Print first line of frame

        # Search for similar items in the vector DB
        embeddings_to_compare = torch.stack([item['embedding'] for item in vector_db])
        cosine_scores = util.cos_sim(frame_embedding, embeddings_to_compare)[0] # Using torch for cosine similarity

        # Get top-k results above the similarity threshold
        top_results_indices = [idx for idx, score in enumerate(cosine_scores) if score > SIMILARITY_THRESHOLD]
        top_results_indices = sorted(top_results_indices, key=lambda idx: cosine_scores[idx], reverse=True)[:k]

        if not top_results_indices:
            print("No relevant context found for this frame above the threshold.")
            continue

        print("Found relevant context:")
        for idx in top_results_indices:
            item = vector_db[idx]
            score = cosine_scores[idx].item() # Get scalar value from tensor

            if item['type'] == 'code':
                if item['id'] not in retrieved_code:
                     retrieved_code[item['id']] = {"item": item, "score": score}
                     ast_features_info = ""
                     # --- Demonstrate AST Feature Value (Simple Example) ---
                     # If the error message suggests an exception, highlight code with try/except
                     if "Error" in error_message or "Exception" in error_message:
                         if item.get('ast_features', {}).get('has_try_except'):
                             ast_features_info += " (Structurally Relevant: Has Try/Except)"
                         elif item.get('ast_features', {}).get('has_if'):
                              ast_features_info += " (Structurally Relevant: Has Conditional)"
                         elif item.get('ast_features', {}).get('has_for') or item.get('ast_features', {}).get('has_while'):
                             ast_features_info += " (Structurally Relevant: Has Loop)"

                     print(f"  - Code Snippet (Score: {score:.4f}): {item['metadata']['file']}::{item['metadata']['function']}{ast_features_info}")
                     # In a real system, you might use these features for re-ranking,
                     # generating a more specific prompt for the LLM, etc.


            elif item['type'] == 'stack_frame':
                 if item['id'] not in retrieved_stack_frames:
                    retrieved_stack_frames[item['id']] = {"item": item, "score": score}
                    print(f"  - Similar Stack Frame (Score: {score:.4f}): {item['content'].splitlines()[0]}") # Print first line


    return retrieved_code.values(), retrieved_stack_frames.values()

# --- Simulate New Error Events ---
# We'll also include a simulated error message now
new_error_1_message = "Database connection failed"
new_error_stack_trace_1 = [
    "File \"api_handler.py\", line 11, in handle_request\n    save_user_to_db(user_info)",
    "File \"user_service.py\", line 14, in save_user_to_db\n    db_connection = connect_db()",
    "File \"database_utils.py\", line 6, in connect_db\n    raise ConnectionError('Database connection failed')" # The erroring frame
]

new_error_2_message = "Input must be a dictionary"
new_error_stack_trace_2 = [
     "File \"api_handler.py\", line 10, in handle_request\n    user_info = process_user_input(request_data)",
     "File \"user_service.py\", line 5, in process_user_input\n    if not isinstance(data, dict):", # The erroring frame
]

new_error_3_message = "Order data must be a dictionary"
new_error_stack_trace_3 = [
     "File \"some_api.py\", line 25, in handle_order_request\n    order_details = process_order_data(request_payload)",
     "File \"order_service.py\", line 8, in process_order_data\n    if not isinstance(data, dict):", # The erroring frame - semantically similar to st_frame_2
]


# --- Run the POC ---

# Need to import torch for sentence-transformers' tensor operations
try:
    import torch
except ImportError:
    print("PyTorch not found. Please install it (`pip install torch`) to run this POC.")
    exit()


print("\n--- Running POC with Error Trace 1 (Database Connection Error) ---")
retrieved_code_1, retrieved_stack_frames_1 = retrieve_context(new_error_stack_trace_1, new_error_1_message)

print("\n--- Retrieved Context for Error Trace 1 ---")
print("Relevant Code Snippets:")
if retrieved_code_1:
    for item_info in retrieved_code_1:
        item = item_info["item"]
        ast_features_info = ""
        if "Error" in new_error_1_message or "Exception" in new_error_1_message:
             if item.get('ast_features', {}).get('has_try_except'):
                 ast_features_info += " (Structurally Relevant: Has Try/Except)"
             elif item.get('ast_features', {}).get('has_if'):
                  ast_features_info += " (Structurally Relevant: Has Conditional)"
             elif item.get('ast_features', {}).get('has_for') or item.get('ast_features', {}).get('has_while'):
                 ast_features_info += " (Structurally Relevant: Has Loop)"

        print(f"- {item['metadata']['file']}::{item['metadata']['function']} (Score: {item_info['score']:.4f}){ast_features_info}")
else:
    print("None found.")

print("\nSemantically Similar Stack Frames:")
if retrieved_stack_frames_1:
    for item_info in retrieved_stack_frames_1:
        item = item_info["item"]
        print(f"- {item['content'].splitlines()[0]} (Score: {item_info['score']:.4f})")
else:
    print("None found.")


print("\n\n--- Running POC with Error Trace 2 (User Input Validation Error) ---")
retrieved_code_2, retrieved_stack_frames_2 = retrieve_context(new_error_stack_trace_2, new_error_2_message)

print("\n--- Retrieved Context for Error Trace 2 ---")
print("Relevant Code Snippets:")
if retrieved_code_2:
    for item_info in retrieved_code_2:
        item = item_info["item"]
        ast_features_info = ""
        if "Error" in new_error_2_message or "Exception" in new_error_2_message:
             if item.get('ast_features', {}).get('has_try_except'):
                 ast_features_info += " (Structurally Relevant: Has Try/Except)"
             elif item.get('ast_features', {}).get('has_if'):
                  ast_features_info += " (Structurally Relevant: Has Conditional)"
             elif item.get('ast_features', {}).get('has_for') or item.get('ast_features', {}).get('has_while'):
                 ast_features_info += " (Structurally Relevant: Has Loop)"
        print(f"- {item['metadata']['file']}::{item['metadata']['function']} (Score: {item_info['score']:.4f}){ast_features_info}")
else:
    print("None found.")

print("\nSemantically Similar Stack Frames:")
if retrieved_stack_frames_2:
    for item_info in retrieved_stack_frames_2:
        item = item_info["item"]
        print(f"- {item['content'].splitlines()[0]} (Score: {item_info['score']:.4f})")
else:
    print("None found.")

print("\n\n--- Running POC with Error Trace 3 (Order Input Validation Error - Semantic Similarity Demo) ---")
retrieved_code_3, retrieved_stack_frames_3 = retrieve_context(new_error_stack_trace_3, new_error_3_message)

print("\n--- Retrieved Context for Error Trace 3 ---")
print("Relevant Code Snippets:")
if retrieved_code_3:
    for item_info in retrieved_code_3:
        item = item_info["item"]
        ast_features_info = ""
        if "Error" in new_error_3_message or "Exception" in new_error_3_message:
             if item.get('ast_features', {}).get('has_try_except'):
                 ast_features_info += " (Structurally Relevant: Has Try/Except)"
             elif item.get('ast_features', {}).get('has_if'):
                  ast_features_info += " (Structurally Relevant: Has Conditional)"
             elif item.get('ast_features', {}).get('has_for') or item.get('ast_features', {}).get('has_while'):
                 ast_features_info += " (Structurally Relevant: Has Loop)"
        print(f"- {item['metadata']['file']}::{item['metadata']['function']} (Score: {item_info['score']:.4f}){ast_features_info}")
else:
    print("None found.")

print("\nSemantically Similar Stack Frames:")
if retrieved_stack_frames_3:
    for item_info in retrieved_stack_frames_3:
        item = item_info["item"]
        print(f"- {item['content'].splitlines()[0]} (Score: {item_info['score']:.4f})")
else:
    print("None found.")
