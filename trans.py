from sentence_transformers import SentenceTransformer, util

# Load a pre-trained Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Additional queries
test_queries = [
    "How does AI impact healthcare?",
    "What are the latest trends in computer vision?",
    "Explain climate change effects on the environment.",
    "Benefits of renewable energy sources.",
    "Applications of machine learning in finance.",
    "History of the internet and web development.",
    "Techniques for improving cybersecurity.",
    "Role of NLP in chatbots.",
    "Recent breakthroughs in robotics.",
    "How does quantum computing work?"
]

# Example additional documents
documents = [
    "AI is revolutionizing healthcare with predictive analytics.",
    "Computer vision enables self-driving cars to recognize objects.",
    "Climate change leads to rising sea levels and extreme weather events.",
    "Renewable energy like solar and wind reduces carbon emissions.",
    "Machine learning helps detect fraud in financial transactions.",
    "The internet started as ARPANET in the late 1960s.",
    "Cybersecurity techniques include encryption and multi-factor authentication.",
    "NLP powers chatbots to understand and respond to human language.",
    "Robotics advancements include humanoid robots and autonomous drones.",
    "Quantum computing uses qubits to perform complex calculations faster."
]

doc_embeddings = model.encode(documents, convert_to_tensor=True)

# Example of running semantic search for multiple queries
for query in test_queries:
    query_embedding = model.encode(query, convert_to_tensor=True)
    results = util.semantic_search(query_embedding, doc_embeddings, top_k=1)
    most_similar_doc = documents[results[0][0]['corpus_id']]
    print(f"Query: {query}\nMost similar document: {most_similar_doc}\n")



query_embedding = model.encode(query)


# Find the most similar document to the query
results = util.semantic_search(query_embedding, doc_embeddings, top_k=1)
most_similar_doc = documents[results[0][0]['corpus_id']]
print(f"Most similar document to the query: \"{most_similar_doc}\"")