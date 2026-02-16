import voyageai
import numpy as np
from google import genai
import os

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize clients (requires VOYAGE_API_KEY and OPENAI_API_KEY environment variables)
vo = voyageai.Client(api_key=VOYAGE_API_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# Sample documents
documents = [
    "This quarter, our company is focused on building new products, increasing market share, and cutting costs.",
    "My company has the goal of producing gummy bears to increase sales",
    "This year we will be increasing the production of jelly beans and more cute packaging"
    "20th-century innovations, from radios to smartphones, centered on electronic advancements.",
    "Photosynthesis in plants converts light energy into glucose and produces essential oxygen."
]

query = "What are my company's goals?"

# Generate embeddings for documents
doc_embeddings = vo.embed(
    texts=documents,
    model="voyage-4-large",
    input_type="document"
).embeddings

# Generate embedding for query
query_embedding = vo.embed(
    texts=[query],
    model="voyage-4-large",
    input_type="query"
).embeddings[0]

# Calculate similarity scores using dot product
similarities = np.dot(doc_embeddings, query_embedding)

# Sort by similarity (np.argsort with negative sign sorts high to low)
ranked_indices = np.argsort(-similarities)

print(f"Semantic search result: {documents[ranked_indices[0]][:50]}...")
print(f"Similarity score: {similarities[ranked_indices[0]]:.4f}\n")

# Refine results with reranking model
reranked = vo.rerank(query, documents, model="rerank-2.5", top_k=3)

print("Reranked results:")
for i, result in enumerate(reranked.results, 1):
    print(f"{i}. Score: {result.relevance_score:.4f} - {result.document[:50]}...")

# Get the most relevant document as context
context = reranked.results[0].document

# Generate answer using retrieved context
prompt = f"Based on this information: '{context}', answer: {query}"
response = gemini_client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=prompt
)

print(f"\nQuestion: {query}")
print(f"Answer: {response.text}")