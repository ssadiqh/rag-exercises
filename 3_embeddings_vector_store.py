"""Exercise 3: Embeddings and Vector Storage
Learn how to embed documents and store them in vector databases for semantic search."""

import numpy as np

print("=" * 70)
print("EXERCISE 3: Embeddings and Vector Storage")
print("=" * 70)

# Step 1: Understanding embeddings
print("\n1. WHAT ARE EMBEDDINGS?")
print("-" * 70)

print("""
Embeddings are numerical representations of text that capture semantic meaning:
  - Convert text into vectors (lists of numbers)
  - Texts with similar meaning have similar vectors
  - Allow semantic similarity search (not just keyword matching)
  - Enable LLMs to understand relationships between concepts

Example:
  - "What is the mobile policy?" → [0.24, -0.18, 0.56, ..., 0.02] (384 dimensions)
  - "Describe phone device rules" → [0.25, -0.19, 0.55, ..., 0.03] (similar!)
  - "What is the weather?" → [0.01, 0.92, -0.34, ..., -0.71] (very different)
""")

# Step 2: Simulate simple embeddings
print("\n2. SIMULATING SIMPLE EMBEDDINGS")
print("-" * 70)

class SimpleEmbedding:
    """Simple embedding using word frequency (for demonstration)"""

    def __init__(self, vocabulary_size=100):
        self.vocabulary_size = vocabulary_size
        self.word_to_idx = {}
        self.idx = 0

    def encode(self, text: str) -> np.ndarray:
        """Convert text to a simple embedding"""
        words = text.lower().split()

        # Create word index mapping
        for word in words:
            if word not in self.word_to_idx:
                if self.idx < self.vocabulary_size:
                    self.word_to_idx[word] = self.idx
                    self.idx += 1

        # Create embedding vector (one-hot encoding)
        embedding = np.zeros(self.vocabulary_size)
        for word in words:
            if word in self.word_to_idx:
                embedding[self.word_to_idx[word]] += 1

        # Normalize
        if np.sum(embedding) > 0:
            embedding = embedding / np.sum(embedding)

        return embedding

    def similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

# Sample texts
texts = [
    "Mobile device policy ensures security and productivity",
    "Employees must protect company mobile devices",
    "Remote work is permitted three days per week",
    "Smoking is prohibited inside company buildings",
    "Business casual dress code is required",
]

embedder = SimpleEmbedding(vocabulary_size=100)

print(f"Encoding {len(texts)} text samples...")
embeddings = [embedder.encode(text) for text in texts]

print(f"\nEmbedding dimensions: {embeddings[0].shape[0]}")
print(f"First embedding shape: {embeddings[0].shape}")
print(f"First embedding (first 10 values): {embeddings[0][:10]}")

# Step 3: Calculate similarity
print("\n\n3. SEMANTIC SIMILARITY SEARCH")
print("-" * 70)

query = "mobile device security"
query_embedding = embedder.encode(query)

print(f"Query: '{query}'")
print(f"\nSimilarity scores to all documents:")

similarities = []
for i, text in enumerate(texts):
    score = embedder.similarity(query_embedding, embeddings[i])
    similarities.append((text, score))
    print(f"  {score:.3f} - {text[:50]}")

# Sort by similarity
similarities.sort(key=lambda x: x[1], reverse=True)
print(f"\nTop result: '{similarities[0][0][:50]}' (score: {similarities[0][1]:.3f})")

# Step 4: Real embedding models
print("\n\n4. REAL EMBEDDING MODELS IN PRODUCTION")
print("-" * 70)

embedding_models = {
    "HuggingFace Models": [
        ("sentence-transformers/all-MiniLM-L6-v2", 384, "Fast, lightweight, good for RAG"),
        ("sentence-transformers/all-mpnet-base-v2", 768, "Better quality, slower"),
        ("sentence-transformers/distilbert-base-multilingual-cased-v2", 768, "Multilingual"),
    ],
    "OpenAI": [
        ("text-embedding-3-small", 1536, "Commercial API"),
        ("text-embedding-3-large", 3072, "Higher quality"),
    ],
    "Local Models": [
        ("ollama/nomic-embed-text", 768, "Open source, local"),
        ("ollama/all-minilm", 384, "Lightweight local"),
    ],
}

for category, models in embedding_models.items():
    print(f"\n{category}:")
    for model, dim, description in models:
        print(f"  - {model}")
        print(f"    Dimensions: {dim}, {description}")

# Step 5: Vector storage concepts
print("\n\n5. VECTOR STORAGE & RETRIEVAL")
print("-" * 70)

print("""
Vector databases are optimized for semantic search:

Common Vector Stores:
  1. Chroma: Simple, open-source, SQLite-based
  2. Pinecone: Cloud-based, fully managed
  3. Weaviate: Open-source, full-featured
  4. Milvus: Scalable, high-performance
  5. Qdrant: Rust-based, production-ready
  6. FAISS: Facebook's efficient similarity search
  7. Elasticsearch: Text + vector search

Storage Process:
  1. Embed each document chunk
  2. Store embedding + metadata
  3. Build index for fast retrieval
  4. Query: embed user query, find nearest neighbors

Retrieval Methods:
  - Exact: Calculate all distances (slow, accurate)
  - Approximate: Use HNSW/IVF indices (fast, approximate)
  - Hybrid: Text + semantic search combined
""")

# Step 6: Simulate vector store
print("\n\n6. SIMULATING A VECTOR STORE")
print("-" * 70)

class SimpleVectorStore:
    """Simple in-memory vector store for demonstration"""

    def __init__(self):
        self.documents = []
        self.embeddings = []
        self.metadata = []

    def add_documents(self, texts: list, embeddings: list, metadata: list = None):
        """Add documents to the store"""
        self.documents.extend(texts)
        self.embeddings.extend(embeddings)

        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{"id": i} for i in range(len(texts))])

    def search(self, query_embedding: np.ndarray, k: int = 3) -> list:
        """Find top k most similar documents"""
        similarities = []

        embedder = SimpleEmbedding()
        for i, emb in enumerate(self.embeddings):
            score = embedder.similarity(query_embedding, emb)
            similarities.append({
                "text": self.documents[i],
                "score": score,
                "metadata": self.metadata[i]
            })

        # Sort by score and return top k
        similarities.sort(key=lambda x: x["score"], reverse=True)
        return similarities[:k]

# Create and populate vector store
vector_store = SimpleVectorStore()
metadata = [
    {"source": "policies.txt", "policy": "Mobile Device Policy"},
    {"source": "policies.txt", "policy": "Mobile Device Policy"},
    {"source": "policies.txt", "policy": "Remote Work Policy"},
    {"source": "policies.txt", "policy": "Smoking Policy"},
    {"source": "policies.txt", "policy": "Dress Code"},
]

vector_store.add_documents(texts, embeddings, metadata)

# Search
query = "mobile phone device rules"
query_emb = embedder.encode(query)
results = vector_store.search(query_emb, k=2)

print(f"Query: '{query}'")
print(f"\nTop 2 results:")
for i, result in enumerate(results, 1):
    print(f"\n  Result {i}:")
    print(f"    Similarity: {result['score']:.3f}")
    print(f"    Policy: {result['metadata']['policy']}")
    print(f"    Text: {result['text'][:50]}...")

# Step 7: Practical workflow
print("\n\n7. COMPLETE RAG WORKFLOW")
print("-" * 70)

print("""
Complete RAG Indexing & Retrieval Flow:

INDEXING (Offline, one-time):
  1. Load documents
  2. Split into chunks
  3. Generate embeddings for each chunk
  4. Store embeddings + metadata in vector DB
  5. Build index for fast search

RETRIEVAL (At query time):
  1. Generate embedding for user query
  2. Search vector store (find similar chunks)
  3. Retrieve top-k chunks + metadata
  4. Pass to LLM as context
  5. LLM generates answer

Example:
  User Query: "What is the mobile policy?"
    ↓
  Embedding: [0.12, -0.45, 0.78, ...]
    ↓
  Vector Search: Find similar chunks
    ↓
  Retrieved Context: [Policy 1 chunk 1, Policy 1 chunk 2, ...]
    ↓
  LLM Prompt: "Context: <chunks>. Question: What is the mobile policy?"
    ↓
  Answer: "The mobile device policy ensures that all employees..."
""")

# Step 8: Summary
print("\n" + "=" * 70)
print("KEY CONCEPTS:")
print("=" * 70)
print("""
1. EMBEDDINGS:
   - Convert text to high-dimensional vectors
   - Capture semantic meaning
   - Enable similarity search
   - Usually 300-3000 dimensions

2. EMBEDDING MODELS:
   - HuggingFace: Popular, open-source, varied quality
   - OpenAI: High quality, API-based
   - Local: Privacy-friendly, Ollama support
   - Choose based on quality/speed/privacy needs

3. VECTOR DATABASES:
   - Store embeddings + metadata
   - Optimized for similarity search
   - Index structures: HNSW, IVF, etc.
   - Various options: Chroma, Pinecone, Weaviate, etc.

4. SIMILARITY METRICS:
   - Cosine similarity: Most common (dot product / norms)
   - Euclidean distance: Less common
   - Manhattan distance: Rarely used
   - Cosine similarity is 0-1 (0=opposite, 1=identical)

5. PRACTICAL CONSIDERATIONS:
   - Embedding quality impacts RAG quality
   - Store metadata for source attribution
   - Index size affects retrieval speed
   - Consider privacy (local vs cloud embeddings)
   - Batch embedding for efficiency

6. OPTIMIZATION:
   - Choose embedding model size/speed trade-off
   - Tune chunk size for embedding quality
   - Use approximate search for large datasets
   - Cache embeddings to avoid recomputation
   - Monitor retrieval quality metrics
""")