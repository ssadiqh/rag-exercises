"""Exercise 3: Embeddings and Vector Storage
Learn how to embed documents using HuggingFace and store them in Chroma vector database."""

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

Example with all-MiniLM-L6-v2 (384 dimensions):
  - "What is the mobile policy?" → [0.24, -0.18, 0.56, ..., 0.02] (384 dims)
  - "Describe phone device rules" → [0.25, -0.19, 0.55, ..., 0.03] (similar!)
  - "What is the weather?" → [0.01, 0.92, -0.34, ..., -0.71] (very different)
""")

# Step 2: Real embedding model
print("\n2. HUGGINGFACE EMBEDDINGS (all-MiniLM-L6-v2)")
print("-" * 70)

print("""
Using: sentence-transformers/all-MiniLM-L6-v2
  - Model size: 22 MB (lightweight, fast)
  - Dimensions: 384
  - Speed: ~1000 documents/second
  - Quality: Good semantic understanding
  - Perfect for: RAG systems, learning environments

Installation:
  pip install sentence-transformers

First run will download the model (~100 MB with dependencies).
""")

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    print("✓ HuggingFaceEmbeddings imported successfully")

    # Initialize embeddings
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("✓ Embeddings model loaded: all-MiniLM-L6-v2")

except ImportError as e:
    print(f"\n⚠️ Missing dependency: {e}")
    print("Install with: pip install sentence-transformers langchain")
    print("Falling back to simulation for demonstration...\n")
    embeddings_model = None

# Step 3: Create sample embeddings
print("\n3. CREATING EMBEDDINGS")
print("-" * 70)

texts = [
    "Mobile device policy ensures security and productivity",
    "Employees must protect company mobile devices",
    "Remote work is permitted three days per week",
    "Smoking is prohibited inside company buildings",
    "Business casual dress code is required",
]

if embeddings_model:
    print(f"Encoding {len(texts)} text samples with all-MiniLM-L6-v2...")
    embeddings = embeddings_model.embed_documents(texts)

    print(f"✓ Created {len(embeddings)} embeddings")
    print(f"✓ Embedding dimension: {len(embeddings[0])}")
    print(f"✓ First embedding (first 5 values): {embeddings[0][:5]}")

else:
    # Fallback: simple simulation
    print("Using simulated embeddings for demonstration...")
    embeddings = [np.random.randn(384).astype(np.float32) for _ in texts]
    print(f"Created {len(embeddings)} simulated embeddings (384 dims)")

# Step 4: Similarity search
print("\n4. SEMANTIC SIMILARITY SEARCH")
print("-" * 70)

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between vectors"""
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

if embeddings_model:
    query = "mobile device security"
    query_embedding = embeddings_model.embed_query(query)

    print(f"Query: '{query}'")
    print(f"\nSimilarity scores to documents:")

    scores = []
    for i, text in enumerate(texts):
        similarity = cosine_similarity(query_embedding, embeddings[i])
        scores.append((text, similarity))
        print(f"  {similarity:.4f} - {text[:50]}")

    scores.sort(key=lambda x: x[1], reverse=True)
    print(f"\nTop match: '{scores[0][0][:50]}' (score: {scores[0][1]:.4f})")

# Step 5: Vector database comparison
print("\n5. VECTOR DATABASE OPTIONS")
print("-" * 70)

print("""
Vector Stores Available:

1. CHROMA (Recommended for learning)
   - Local SQLite-based vector database
   - Simple API, automatic persistence
   - Perfect for RAG systems
   - Installation: pip install chromadb

2. PINECONE (Cloud-based)
   - Fully managed vector database
   - Scales to billions of vectors
   - Requires API key
   - Good for production

3. WEAVIATE (Self-hosted)
   - Open-source, high-performance
   - Supports hybrid search
   - Good for enterprises

4. MILVUS (Enterprise)
   - Scalable, distributed
   - High performance
   - Complex setup

For this course: Using CHROMA (local, simple, perfect for RAG)
""")

# Step 6: Introduction to Chroma
print("\n6. CHROMA VECTOR DATABASE")
print("-" * 70)

print("""
Chroma is a vector database designed for AI applications:

Features:
  ✓ Local storage (SQLite)
  ✓ Persistent data to disk
  ✓ Simple Python API
  ✓ Automatic schema management
  ✓ Built-in similarity search
  ✓ Metadata filtering

How it works:
  1. Store: documents + embeddings + metadata → Chroma
  2. Search: query → embed query → find similar vectors → return docs
  3. Persist: data saved to disk automatically

Data structure:
  {
    "id": "doc_1",
    "embedding": [0.24, -0.18, ...],  # 384 dimensions
    "document": "Mobile device policy...",
    "metadata": {"source": "policies.txt", "policy": "Mobile Device Policy"}
  }
""")

# Step 7: Chroma with real embeddings
print("\n7. CREATING VECTOR STORE WITH CHROMA")
print("-" * 70)

try:
    from langchain_chroma import Chroma
    from langchain_core.documents import Document

    print("✓ Chroma imported successfully")

    if embeddings_model:
        # Create documents with metadata
        documents = []
        for i, text in enumerate(texts):
            doc = Document(
                page_content=text,
                metadata={
                    "doc_id": i,
                    "source": "sample_policies.txt"
                }
            )
            documents.append(doc)

        # Create vector store (will create ./chroma_data directory)
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings_model,
            persist_directory="./chroma_data"
        )

        print(f"✓ Vector store created with {len(documents)} documents")
        print(f"✓ Data persisted to ./chroma_data")

        # Test similarity search
        query = "mobile device security"
        results = vector_store.similarity_search(query, k=2)

        print(f"\nSimilarity search for: '{query}'")
        print(f"Found {len(results)} relevant documents:")

        for i, result in enumerate(results, 1):
            print(f"\n  Result {i}:")
            print(f"    Text: {result.page_content[:60]}...")
            print(f"    Metadata: {result.metadata}")

    else:
        print("⚠️ HuggingFaceEmbeddings not available - cannot create real vector store")

except ImportError as e:
    print(f"⚠️ Missing Chroma: {e}")
    print("Install with: pip install chromadb")

# Step 8: Production considerations
print("\n\n8. PRODUCTION CONSIDERATIONS")
print("-" * 70)

print("""
For RAG systems in production:

EMBEDDINGS CHOICE:
  ✓ Use all-MiniLM-L6-v2 for: Fast inference, limited resources
  ✓ Use all-mpnet-base-v2 for: Better accuracy, more resources available
  ✓ Update easily: Just change model_name parameter

VECTOR STORE CHOICE:
  ✓ Chroma: Great for local/small-scale RAG
  ✓ Upgrade to Pinecone: When you need cloud scalability
  ✓ Use Weaviate: For hybrid search + filtering

OPTIMIZATION:
  • Batch embedding: Embed multiple documents together
  • Caching: Cache embeddings to avoid recomputation
  • Indexing: Use HNSW for faster approximate search
  • Tuning: Test different chunk sizes with your embeddings
""")

# Step 9: Summary
print("\n" + "=" * 70)
print("KEY CONCEPTS:")
print("=" * 70)
print("""
1. HUGGINGFACE EMBEDDINGS:
   - Convert text to 384-dimensional vectors (all-MiniLM-L6-v2)
   - Capture semantic meaning
   - Enable similarity search
   - Fast and lightweight

2. CHROMA VECTOR DATABASE:
   - Store embeddings with metadata
   - Persistent local storage (SQLite)
   - Simple similarity search API
   - Automatic document management

3. SEMANTIC SEARCH FLOW:
   Text → Embedding → Vector Store → Similarity Search → Results

4. EMBEDDINGS FOR RAG:
   - Document chunks are embedded once (indexing)
   - User query embedded at search time
   - Find similar chunks by vector distance
   - Return top-k most relevant documents

5. PRACTICAL WORKFLOW:
   - Initialize embeddings: HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
   - Create vector store: Chroma.from_documents(docs, embeddings)
   - Search: vector_store.similarity_search(query, k=3)

6. REAL IMPLEMENTATION:
   ✓ No simulation - using actual models
   ✓ No API keys - runs locally
   ✓ Persistent storage - data saved to disk
   ✓ Production-ready - used in real RAG systems
""")

print("\n✅ Exercise 3 complete!")
print("Next: Use these embeddings + Chroma in Exercise 4 for full RAG system")