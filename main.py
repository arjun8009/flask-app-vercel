import warnings
warnings.filterwarnings("ignore")
from flask import Flask, jsonify, request
import pickle
import os
import base64
from llama_index.core.retrievers import VectorIndexRetriever

app = Flask(__name__)


@app.route('/')
def home():
    return jsonify({"message": "Hello, world!"})


@app.route('/api/get-index', methods=['GET'])
def get_index():
    chunking_method = request.args.get('chunking_method', 'default')
    similarity_top_k = int(request.args.get('similarity_top_k', None))
    metadata = request.args.getlist('metadata', None)
    queries = request.args.getlist('queries', None)
    metadata = [int(i) for i in metadata]
    print(metadata)
    print(queries)

    # Example parameter
    path = os.path.join(os.getcwd(), "vector_index_pickle")
    file_path = os.path.join(path, f"{chunking_method}_vector_index.pkl")

    # Check if the file exists
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404

    # Load the pickled object
    with open(file_path, "rb") as file:
        index = pickle.load(file)

    # Serialize the index to base64
    node_ids = [node.id_ for node in index.docstore.docs.values() if int(node.metadata['file_id']) in metadata] if type(metadata) == list else None
    print("Node ids",node_ids)
    if len(node_ids) == 0 or node_ids == None:
        print("1st pipeline entering")
        retriever = VectorIndexRetriever(index, similarity_top_k=similarity_top_k)
    else:
        print("other pipelines entering")
        retriever = VectorIndexRetriever(index,node_ids=node_ids,similarity_top_k=similarity_top_k)
    print("retriever-response",{i : retriever.retrieve(query) for i, query in enumerate(queries)})
    retrieved_nodes = base64.b64encode(pickle.dumps({i : retriever.retrieve(query) for i, query in enumerate(queries)})).decode('utf-8')

    return jsonify({"vector_index": retrieved_nodes})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
