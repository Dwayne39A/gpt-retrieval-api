import os
import json
import openai
from pinecone import Pinecone  # Corrected import
from flask import Flask, request, jsonify

# Load API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

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
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response["data"][0]["embedding"]

# API Endpoint: Retrieve relevant business knowledge
@app.route('/retrieve', methods=['POST'])
def retrieve():
    data = request.json
    query = data.get("query", "")

    # Convert query to embedding
    query_embedding = get_embedding(query)

    # Search Pinecone
    results = index.query(vector=query_embedding, top_k=3, include_metadata=True)  # Fix applied

    # Extract relevant documents
    retrieved_docs = [match["metadata"]["text"] for match in results["matches"]]

    return jsonify({"retrieved_text": retrieved_docs})

# Run the API
if __name__ == "__main__":
    app.run(debug=True)
