import os
import json
import openai
from pinecone import Pinecone  # Corrected import
from flask import Flask, request, jsonify

# Load API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

# Initialize OpenAI Client
openai.api_key = OPENAI_API_KEY  # This is now handled globally

# Initialize Pinecone client (Fix applied here)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("knowledge-base")  # Corrected usage

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "API is running. Use POST /retrieve endpoint to query."

# Function to get embeddings from OpenAI
def get_embedding(text):
    response = openai.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding  # Extract the embedding correctly

# API Endpoint: Retrieve relevant business knowledge
@app.route('/retrieve', methods=['POST'])
def retrieve():
    data = request.json
    query = data.get("query", "")

    if not query:
        return jsonify({"error": "No query provided"}), 400

    try:
        # Convert query to embedding
        query_embedding = get_embedding(query)

        # Search Pinecone
        results = index.query(vector=query_embedding, top_k=3, include_metadata=True)

        # Extract relevant documents
        retrieved_docs = [match["metadata"]["text"] for match in results["matches"]]
        
        return jsonify({"retrieved_text": retrieved_docs})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the API
if __name__ == "__main__":
    app.run(debug=True)
