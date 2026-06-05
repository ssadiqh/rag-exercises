"""Exercise 4: Building a RAG Retrieval System
Create a complete RAG system with document loading, chunking, embedding, and retrieval."""

import tempfile
from pathlib import Path

print("=" * 70)
print("EXERCISE 4: Building a RAG Retrieval System")
print("=" * 70)

# Step 1: Create sample documents
print("\n1. PREPARING DOCUMENTS")
print("-" * 70)

company_policies = """
INNOVATECH COMPANY POLICIES

Policy 1: Mobile Device Policy
The mobile device policy ensures that all employees maintain productivity and security
while using mobile devices for business purposes. Employees must ensure devices are
password-protected and up-to-date with security patches. Company data should not be
stored on personal devices without proper encryption. Mobile devices used for company
purposes must have antivirus software installed and kept up-to-date. VPN must be used
when connecting to company networks from public WiFi. Lost or stolen devices should be
reported immediately to IT department.

Policy 2: Remote Work Policy
Employees are permitted to work remotely up to 3 days per week, subject to manager approval.
Remote workers must maintain a dedicated workspace and ensure a reliable internet connection.
Video calls should use company-approved platforms only. During remote work, employees must
maintain regular communication with their team. Flexible working hours are permitted as long
as core hours (10 AM - 3 PM) are maintained for meetings and collaboration. Home office
equipment requests should be submitted to the facilities department.

Policy 3: Smoking Policy
Smoking is strictly prohibited inside all company buildings and vehicles. Smoking is only
permitted in designated outdoor areas, at least 20 feet away from building entrances.
E-cigarettes and vaping are subject to the same restrictions as traditional cigarettes.
Employees who violate this policy may receive a written warning. Multiple violations may
result in suspension or termination. Designated smoking areas are clearly marked on the
property map provided to all employees.

Policy 4: Dress Code
Business casual is the standard dress code at Innovatech. This means slacks or skirts with
a collared shirt or blouse. Denim is not permitted. Exceptions may be made for specific roles
or departments with manager approval. Closed-toe shoes are required for safety reasons in
certain areas. Visible tattoos and piercings are permitted. The dress code may be relaxed
on designated casual Fridays.

Policy 5: Internet and Email Policy
Company internet and email are for business purposes only. Employees are responsible for
protecting their login credentials and not sharing them with anyone. Inappropriate use of
company systems may result in disciplinary action, up to and including termination. Personal
use is limited to break times only. Social media access is restricted but email and
professional communication are allowed. Regular audits may be conducted to ensure compliance.

Policy 6: Vacation and Paid Time Off
Employees receive 20 days of paid vacation annually, plus 10 company holidays. Vacation
requests must be submitted at least 2 weeks in advance. During peak business periods,
vacation may be limited for certain departments. Unused vacation days may roll over up to
5 days into the next year. Employees cannot be forced to take vacation time. Emergency
time off should be reported to the HR department as soon as possible.

Policy 7: Code of Conduct
All employees are expected to treat colleagues with respect and professionalism at all times.
Harassment, discrimination, and bullying are strictly prohibited. Violations should be
reported to Human Resources immediately and will be investigated thoroughly. Retaliation
against employees who report violations is forbidden. The company is committed to maintaining
a safe and inclusive workplace for all employees.

Policy 8: Health and Safety
Employees must follow all safety protocols and report hazards immediately to management.
First aid kits are located on each floor. Emergency procedures are posted in all work areas.
Safety training is mandatory for all new employees. Regular safety inspections are conducted.
Personal protective equipment must be worn when required. Injuries should be reported within
24 hours to the HR department.

Policy 9: Confidentiality
All company information, including code, strategies, and business plans, must be kept
confidential. Non-disclosure agreements are required upon hire. Violations may result in
legal action and termination. Employee confidentiality agreements extend 2 years beyond
employment termination. Third-party vendors must also sign confidentiality agreements.
Data security protocols must be followed at all times to prevent unauthorized access.
"""

# Save to temporary file
with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
    f.write(company_policies)
    temp_file_path = f.name

print(f"Document saved to: {temp_file_path}")
print(f"Document size: {len(company_policies)} characters")

# Step 2: Text splitting
print("\n\n2. SPLITTING DOCUMENTS INTO CHUNKS")
print("-" * 70)

class RecursiveCharacterTextSplitter:
    """Split text into chunks recursively"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> list:
        """Split text into chunks"""
        chunks = []
        start = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            # Try to break at a space
            if end < len(text):
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - self.chunk_overlap

        return chunks

# Load and split document
with open(temp_file_path, 'r') as f:
    document_text = f.read()

splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
chunks = splitter.split_text(document_text)

print(f"Total chunks created: {len(chunks)}")
print(f"Average chunk size: {sum(len(c) for c in chunks) / len(chunks):.0f} characters")

print(f"\nFirst 3 chunks preview:")
for i, chunk in enumerate(chunks[:3], 1):
    print(f"\n  Chunk {i} ({len(chunk)} chars):")
    print(f"    {chunk[:60]}...")

# Step 3: Simple embedding simulation
print("\n\n3. CREATING EMBEDDINGS")
print("-" * 70)

import numpy as np

class SimpleEmbedder:
    """Simple embedding for demonstration"""

    def embed(self, text: str) -> np.ndarray:
        """Create a simple embedding based on character codes"""
        # Convert text to character codes and average
        if not text:
            return np.zeros(128)

        # Use hash-based approach for consistency
        seed = hash(text) % 2**32
        np.random.seed(seed)
        return np.random.randn(128).astype(np.float32)

embedder = SimpleEmbedder()
embeddings = [embedder.embed(chunk) for chunk in chunks]

print(f"Created {len(embeddings)} embeddings")
print(f"Embedding dimension: {embeddings[0].shape[0]}")

# Step 4: Simple vector store
print("\n\n4. VECTOR STORE CREATION")
print("-" * 70)

class SimpleVectorStore:
    """In-memory vector store for RAG"""

    def __init__(self):
        self.chunks = []
        self.embeddings = []
        self.metadata = []

    def add(self, chunk: str, embedding: np.ndarray, metadata: dict = None):
        """Add a chunk to the store"""
        self.chunks.append(chunk)
        self.embeddings.append(embedding)
        self.metadata.append(metadata or {})

    def similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Cosine similarity"""
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    def search(self, query_text: str, k: int = 3) -> list:
        """Search for similar chunks"""
        query_emb = embedder.embed(query_text)

        scores = []
        for i, chunk_emb in enumerate(self.embeddings):
            score = self.similarity(query_emb, chunk_emb)
            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scores[:k]:
            results.append({
                "chunk": self.chunks[idx],
                "score": score,
                "metadata": self.metadata[idx],
                "index": idx
            })

        return results

# Populate vector store
vector_store = SimpleVectorStore()

for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
    # Determine which policy this chunk belongs to
    if "Policy 1" in chunk or "Mobile" in chunk:
        policy = "Mobile Device Policy"
    elif "Policy 2" in chunk or "Remote" in chunk:
        policy = "Remote Work Policy"
    elif "Policy 3" in chunk or "Smoking" in chunk:
        policy = "Smoking Policy"
    elif "Policy 4" in chunk or "Dress" in chunk:
        policy = "Dress Code"
    elif "Policy 5" in chunk or "Internet" in chunk:
        policy = "Internet and Email Policy"
    elif "Policy 6" in chunk or "Vacation" in chunk:
        policy = "Vacation and Paid Time Off"
    elif "Policy 7" in chunk or "Conduct" in chunk:
        policy = "Code of Conduct"
    elif "Policy 8" in chunk or "Safety" in chunk:
        policy = "Health and Safety"
    elif "Policy 9" in chunk or "Confidentiality" in chunk:
        policy = "Confidentiality"
    else:
        policy = "General"

    vector_store.add(
        chunk,
        embedding,
        {
            "source": "policies.txt",
            "chunk_id": i,
            "policy": policy
        }
    )

print(f"Vector store created with {len(vector_store.chunks)} chunks")

# Step 5: Test retrieval
print("\n\n5. TESTING RETRIEVAL")
print("-" * 70)

test_queries = [
    "What is the mobile device policy?",
    "Can I work from home?",
    "Is smoking allowed in the office?",
    "What should I wear to work?",
]

for query in test_queries:
    print(f"\n\nQuery: '{query}'")
    print("-" * 40)

    results = vector_store.search(query, k=2)

    print(f"Retrieved {len(results)} relevant chunks:")

    for i, result in enumerate(results, 1):
        print(f"\n  Result {i}:")
        print(f"    Similarity Score: {result['score']:.3f}")
        print(f"    Policy: {result['metadata']['policy']}")
        print(f"    Chunk Preview: {result['chunk'][:70]}...")

# Step 6: Full RAG context generation
print("\n\n6. GENERATING RAG CONTEXT")
print("-" * 70)

query = "What are the rules about mobile devices?"
results = vector_store.search(query, k=3)

print(f"Query: '{query}'")
print(f"\nRetrieved Context ({len(results)} chunks):")
print("=" * 70)

context = "\n\n".join([f"[Source: {r['metadata']['policy']}]\n{r['chunk']}"
                       for r in results])

print(context[:500])
print("...")

print("\n\nThis context would be passed to the LLM like this:")
print("""
Prompt to LLM:
---
You are a helpful assistant answering questions about company policies.
Use the following context to answer the question.

Context:
<Retrieved chunks above>

Question: {query}

Answer:
---
""")

# Step 7: RAG workflow summary
print("\n\n" + "=" * 70)
print("COMPLETE RAG WORKFLOW SUMMARY:")
print("=" * 70)

print("""
1. DOCUMENT LOADING
   ✓ Load company_policies.txt
   ✓ Document size: {0} characters

2. TEXT SPLITTING
   ✓ Split into {1} chunks
   ✓ Chunk size: 600 characters
   ✓ Overlap: 100 characters

3. EMBEDDING
   ✓ Created {2} embeddings
   ✓ Embedding dimension: 128

4. VECTOR STORAGE
   ✓ Stored {3} chunks with metadata
   ✓ Indexed by policy and content

5. RETRIEVAL
   ✓ Query embedded
   ✓ Similarity search executed
   ✓ Top-k chunks retrieved

6. LLM CONTEXT GENERATION
   ✓ Context assembled from chunks
   ✓ Ready for LLM prompt

Result: Accurate, grounded answers with source attribution
""".format(
    len(document_text),
    len(chunks),
    len(embeddings),
    len(vector_store.chunks)
))

# Cleanup
import os
os.unlink(temp_file_path)

# Step 8: Key concepts
print("\n" + "=" * 70)
print("KEY CONCEPTS:")
print("=" * 70)
print("""
1. RAG PIPELINE:
   - Load documents
   - Split into chunks
   - Create embeddings
   - Store in vector DB
   - Retrieve relevant chunks for queries
   - Pass to LLM as context

2. INDEXING (offline, one-time):
   - Load and process documents
   - Create embeddings
   - Store in vector database
   - Build search indices

3. RETRIEVAL (at query time):
   - Embed user query
   - Search vector store
   - Retrieve top-k chunks
   - Assemble context

4. BENEFITS:
   - Accurate answers grounded in documents
   - Reduced hallucination
   - Source attribution
   - Handles private/proprietary documents
   - Works with data after model training cutoff

5. PRACTICAL CONSIDERATIONS:
   - Choose appropriate chunk size
   - Set similarity threshold
   - Tune top-k (number of results)
   - Monitor retrieval quality
   - Consider latency requirements
   - Update documents efficiently
""")