import warnings
warnings.filterwarnings("ignore")
from flask import Flask, jsonify, request
import pickle
import os
import base64
from llama_index.core.retrievers import VectorIndexRetriever
from functools import lru_cache
import threading
import gc

app = Flask(__name__)

# Global cache for loaded indices
INDEX_CACHE = {}
CACHE_LOCK = threading.Lock()

@lru_cache(maxsize=10)  # Cache up to 10 different chunking methods
def load_index(chunking_method):
    """Load and cache vector index with LRU eviction"""
    with CACHE_LOCK:
        if chunking_method in INDEX_CACHE:
            return INDEX_CACHE[chunking_method]
        
        path = os.path.join(os.getcwd(), "vector_index_pickle")
        file_path = os.path.join(path, f"{chunking_method}_vector_index.pkl")
        
        if not os.path.exists(file_path):
            return None
        
        with open(file_path, "rb") as file:
            index = pickle.load(file)
        
        # Store in cache
        INDEX_CACHE[chunking_method] = index
        return index

@lru_cache(maxsize=100)  # Cache retriever objects
def get_cached_retriever(chunking_method, similarity_top_k, node_ids_tuple=None):
    """Create and cache retriever objects"""
    index = load_index(chunking_method)
    if index is None:
        return None
    
    if node_ids_tuple:
        node_ids = list(node_ids_tuple)
        return VectorIndexRetriever(index, node_ids=node_ids, similarity_top_k=similarity_top_k)
    else:
        return VectorIndexRetriever(index, similarity_top_k=similarity_top_k)

def filter_node_ids(index, metadata):
    """Optimized node ID filtering"""
    if not metadata:
        return None
    
    metadata_set = set(metadata)  # Use set for O(1) lookup
    node_ids = [
        node.id_ for node in index.docstore.docs.values() 
        if int(node.metadata['file_id']) in metadata_set
    ]
    return node_ids if node_ids else None

@app.route('/')
def home():
    return jsonify({"message": "Hello, world!"})

@app.route('/api/get-index', methods=['GET'])
def get_index():
    try:
        # Parse parameters
        chunking_method = request.args.get('chunking_method', 'default')
        similarity_top_k = int(request.args.get('similarity_top_k', 10))
        metadata = request.args.getlist('metadata')
        queries = [q for q in request.args.getlist('queries') if q.strip()]
        
        if not queries:
            return jsonify({"error": "No valid queries provided"}), 400
        
        # Convert metadata to integers
        metadata = [int(m) for m in metadata]
        print(f"Metadata: {metadata}, Queries: {len(queries)} items")

        # Load index from cache
        index = load_index(chunking_method)
        if index is None:
            return jsonify({"error": "Index file not found"}), 404

        # Get filtered node IDs if metadata provided
        node_ids = filter_node_ids(index, metadata) if metadata else None
        print(f"Filtered node IDs: {len(node_ids) if node_ids else 0}")

        # Create cache key for retriever
        node_ids_tuple = tuple(sorted(node_ids)) if node_ids else None
        
        # Get cached retriever
        retriever = get_cached_retriever(chunking_method, similarity_top_k, node_ids_tuple)
        if retriever is None:
            return jsonify({"error": "Failed to create retriever"}), 500

        # Process queries in batch
        retrieved_results = {}
        for i, query in enumerate(queries):
            try:
                retrieved_results[i] = retriever.retrieve(query)
            except Exception as query_error:
                print(f"Error processing query {i}: {query_error}")
                retrieved_results[i] = []

        # Serialize results
        serialized_results = base64.b64encode(pickle.dumps(retrieved_results)).decode('utf-8')
        
        # Trigger garbage collection to free memory
        gc.collect()
        
        return jsonify({"vector_index": serialized_results})
        
    except ValueError as ve:
        return jsonify({"error": f"Invalid parameter: {str(ve)}"}), 400
    except Exception as e:
        print(f"Unexpected error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/clear-cache', methods=['POST'])
def clear_cache():
    """Endpoint to clear caches when memory is low"""
    with CACHE_LOCK:
        INDEX_CACHE.clear()
    load_index.cache_clear()
    get_cached_retriever.cache_clear()
    gc.collect()
    return jsonify({"message": "Cache cleared successfully"})

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "cached_indices": len(INDEX_CACHE),
        "retriever_cache_info": get_cached_retriever.cache_info()._asdict()
    })

if __name__ == '__main__':
    # Production-ready configuration
    app.run(
        host='0.0.0.0', 
        port=8080,
        threaded=True,  # Enable threading for better concurrency
        debug=False     # Disable debug mode for production
    )