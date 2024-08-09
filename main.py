from flask import Flask, jsonify, request
import pickle
import os
import base64

app = Flask(__name__)


@app.route('/')
def home():
    return jsonify({"message": "Hello, world!"})


@app.route('/api/get-index', methods=['GET'])
def get_index():
    chunking_method = request.args.get('chunking_method', 'default')
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
    index_serialized = base64.b64encode(index).decode('utf-8')

    return jsonify({"vector_index": index_serialized})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
