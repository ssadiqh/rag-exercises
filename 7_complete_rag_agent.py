"""Exercise 7: Complete RAG Agent
Build a full-featured RAG agent combining documents, retrieval, prompts, and memory."""

import tempfile
import numpy as np

print("=" * 70)
print("EXERCISE 7: Complete RAG Agent with Memory and Prompts")
print("=" * 70)

# Step 1: Load and prepare documents
print("\n1. LOADING AND PREPARING DOCUMENTS")
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

# Save to temp file
with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
    f.write(company_policies)
    doc_path = f.name

print(f"Document loaded: {len(company_policies)} characters, 9 policies")

# Step 2: Text splitting
print("\n2. SPLITTING DOCUMENTS")
print("-" * 70)

class RecursiveCharacterTextSplitter:
    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> list:
        chunks = []
        start = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            if end < len(text):
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - self.chunk_overlap

        return chunks

splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
chunks = splitter.split_text(company_policies)
print(f"Split into {len(chunks)} chunks")

# Step 3: Create embeddings
print("\n3. CREATING EMBEDDINGS")
print("-" * 70)

class SimpleEmbedder:
    def embed(self, text: str) -> np.ndarray:
        seed = hash(text) % 2**32
        np.random.seed(seed)
        return np.random.randn(128).astype(np.float32)

    def similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

embedder = SimpleEmbedder()
embeddings = [embedder.embed(chunk) for chunk in chunks]
print(f"Created {len(embeddings)} embeddings (dimension: 128)")

# Step 4: Vector store
print("\n4. VECTOR STORE")
print("-" * 70)

class VectorStore:
    def __init__(self):
        self.chunks = []
        self.embeddings = []
        self.metadata = []

    def add(self, chunk: str, embedding: np.ndarray, metadata: dict = None):
        self.chunks.append(chunk)
        self.embeddings.append(embedding)
        self.metadata.append(metadata or {})

    def search(self, query: str, k: int = 3):
        query_emb = embedder.embed(query)
        scores = [(i, embedder.similarity(query_emb, emb))
                  for i, emb in enumerate(self.embeddings)]
        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scores[:k]:
            results.append({
                "chunk": self.chunks[idx],
                "score": score,
                "metadata": self.metadata[idx]
            })
        return results

# Populate vector store
vector_store = VectorStore()

policy_mapping = {
    "Mobile": "Mobile Device Policy",
    "Policy 1": "Mobile Device Policy",
    "Remote": "Remote Work Policy",
    "Policy 2": "Remote Work Policy",
    "Smoking": "Smoking Policy",
    "Policy 3": "Smoking Policy",
    "Dress": "Dress Code",
    "Policy 4": "Dress Code",
    "Internet": "Internet and Email Policy",
    "Policy 5": "Internet and Email Policy",
    "Vacation": "Vacation and Paid Time Off",
    "Policy 6": "Vacation and Paid Time Off",
    "Conduct": "Code of Conduct",
    "Policy 7": "Code of Conduct",
    "Safety": "Health and Safety",
    "Policy 8": "Health and Safety",
    "Confidentiality": "Confidentiality",
    "Policy 9": "Confidentiality",
}

for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
    policy = "General"
    for keyword, policy_name in policy_mapping.items():
        if keyword in chunk:
            policy = policy_name
            break

    vector_store.add(chunk, emb, {"chunk_id": i, "policy": policy, "source": "policies.txt"})

print(f"Vector store ready with {len(vector_store.chunks)} indexed chunks")

# Step 5: Conversation memory
print("\n5. CONVERSATION MEMORY")
print("-" * 70)

class ConversationMemory:
    def __init__(self):
        self.messages = []

    def add_user_message(self, msg: str):
        self.messages.append({"role": "user", "content": msg})

    def add_assistant_message(self, msg: str):
        self.messages.append({"role": "assistant", "content": msg})

    def get_history(self) -> str:
        if not self.messages:
            return ""
        lines = [f"{m['role'].upper()}: {m['content']}" for m in self.messages]
        return "\n".join(lines)

    def clear(self):
        self.messages = []

memory = ConversationMemory()
print("Conversation memory initialized")

# Step 6: Prompt template
print("\n6. PROMPT TEMPLATE")
print("-" * 70)

RAG_PROMPT_TEMPLATE = """You are a helpful assistant answering questions about Innovatech company policies.

INSTRUCTIONS:
1. Use ONLY the provided context to answer questions
2. If the context doesn't contain the answer, say "I don't have information about that"
3. Be concise but complete
4. Reference which policy you're citing

{chat_history}
CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

class PromptTemplate:
    def __init__(self, template: str):
        self.template = template

    def format(self, **kwargs) -> str:
        result = self.template
        for key, value in kwargs.items():
            result = result.replace(f"{{{key}}}", str(value))
        return result

prompt_template = PromptTemplate(RAG_PROMPT_TEMPLATE)
print("Prompt template ready")

# Step 7: Main RAG Agent
print("\n7. COMPLETE RAG AGENT")
print("-" * 70)

class RAGAgent:
    def __init__(self, vector_store, memory, embedder, prompt_template):
        self.vector_store = vector_store
        self.memory = memory
        self.embedder = embedder
        self.prompt_template = prompt_template
        self.turn_count = 0

    def answer(self, query: str, verbose: bool = True) -> str:
        self.turn_count += 1

        if verbose:
            print(f"\n--- Turn {self.turn_count} ---")
            print(f"User: {query}")

        # 1. Retrieve context
        results = self.vector_store.search(query, k=3)
        context = "\n\n".join([
            f"[{r['metadata']['policy']}] {r['chunk'][:200]}..."
            for r in results
        ])

        # 2. Get chat history
        chat_history = memory.get_history()
        if chat_history:
            chat_history = f"PREVIOUS CONVERSATION:\n{chat_history}\n\n"

        # 3. Format prompt
        formatted_prompt = self.prompt_template.format(
            chat_history=chat_history,
            context=context,
            question=query
        )

        # 4. Simulate LLM response
        # In real implementation, this would call an LLM API
        answer = self._simulate_llm_response(query, context)

        if verbose:
            print(f"Assistant: {answer}")
            print(f"Sources: {', '.join([r['metadata']['policy'] for r in results])}")

        # 5. Store in memory
        self.memory.add_user_message(query)
        self.memory.add_assistant_message(answer)

        return answer

    def _simulate_llm_response(self, query: str, context: str) -> str:
        """Simulate LLM response for demonstration"""
        if "mobile" in query.lower():
            return "The mobile device policy requires password protection and security patches. Company data needs encryption, and VPN is required on public WiFi. Lost devices should be reported immediately to IT."

        elif "remote" in query.lower():
            return "Remote work is permitted up to 3 days per week with manager approval. Employees must maintain core hours (10 AM - 3 PM) and use approved platforms for video calls."

        elif "smoke" in query.lower() or "smoking" in query.lower():
            return "Smoking is strictly prohibited inside buildings and vehicles. It's only allowed in designated outdoor areas, at least 20 feet away from entrances."

        elif "dress" in query.lower():
            return "Business casual is the required dress code, meaning slacks/skirts with a collared shirt or blouse. Denim is not permitted, but exceptions may be approved by managers."

        elif "vacation" in query.lower() or "time off" in query.lower():
            return "Employees receive 20 days of paid vacation annually plus 10 company holidays. Requests must be submitted 2 weeks in advance."

        elif "security" in query.lower() or "data" in query.lower():
            return "All company information must be kept confidential. Non-disclosure agreements are required upon hire. Data security protocols must always be followed."

        else:
            return "I can help answer questions about company policies. Please ask about specific policies like mobile devices, remote work, smoking, dress code, vacation, or confidentiality."

# Step 8: Demo conversations
print("\n8. MULTI-TURN CONVERSATION DEMO")
print("=" * 70)

agent = RAGAgent(vector_store, memory, embedder, prompt_template)

demo_queries = [
    "What is the mobile device policy?",
    "Can I store company data on my personal phone?",
    "What if I lose it?",
    "Can I work from home?",
    "How many vacation days do I get?",
]

for query in demo_queries:
    agent.answer(query)

# Step 9: Show conversation memory
print("\n\n9. CONVERSATION MEMORY SUMMARY")
print("=" * 70)
print(memory.get_history())

# Step 10: Complete workflow visualization
print("\n\n" + "=" * 70)
print("COMPLETE RAG AGENT WORKFLOW")
print("=" * 70)

print(f"""
INPUT: User Query
  ↓
RETRIEVAL: Vector Store Search
  - Embed query
  - Find similar chunks
  - Retrieve top-k with metadata
  - {len(results)} chunks retrieved
  ↓
CONTEXT ASSEMBLY: Format Retrieved Context
  - Combine chunks
  - Add policy references
  - Format for LLM
  ↓
MEMORY: Get Conversation History
  - {len(memory.messages)} messages in history
  - Include previous exchanges
  - Maintain context
  ↓
PROMPT FORMATTING: Create LLM Prompt
  - Include system instructions
  - Add conversation history
  - Embed retrieved context
  - Add user question
  ↓
LLM GENERATION: Get Response
  - Call LLM with formatted prompt
  - Generate answer based on context
  - Constrained by instructions
  ↓
OUTPUT: Return Answer
  - Provide user-friendly response
  - Include source attribution
  - Store in conversation memory
  ↓
RESULT: Accurate, Grounded, Traceable Answer
""")

# Step 11: Key metrics
print("\n11. RAG SYSTEM METRICS")
print("-" * 70)

print(f"""
System Performance:
  - Documents indexed: 1
  - Total chunks: {len(vector_store.chunks)}
  - Embedding dimension: 128
  - Retrieval top-k: 3
  - Conversation turns: {agent.turn_count}
  - Memory messages: {len(memory.messages)}

Quality Metrics:
  - Context relevance: Depends on retrieval (vector similarity)
  - Answer accuracy: Depends on LLM and retrieved context
  - Source attribution: Included from metadata
  - Token efficiency: Optimized with chunking and windowing

Improvements for Production:
  - Use real embedding model (HuggingFace sentence-transformers)
  - Use real LLM (Ollama, OpenAI, etc.)
  - Add Chroma or Pinecone for vector storage
  - Implement actual LLM API calls
  - Add error handling and validation
  - Monitor retrieval quality
  - Implement logging and analytics
""")

# Cleanup
import os
os.unlink(doc_path)

# Step 12: Summary
print("\n\n" + "=" * 70)
print("KEY CONCEPTS:")
print("=" * 70)
print("""
1. COMPLETE RAG PIPELINE:
   ✓ Document loading
   ✓ Text splitting
   ✓ Embedding creation
   ✓ Vector storage
   ✓ Semantic retrieval
   ✓ Prompt templating
   ✓ Conversation memory
   ✓ LLM integration

2. DATA FLOW:
   Query → Embed → Search → Retrieve Context
   Context + Memory + Prompt → LLM → Answer
   Answer + Query → Store in Memory

3. KEY COMPONENTS:
   - Vector Store: Index and search documents
   - Embedder: Convert text to vectors
   - Memory: Track conversation history
   - Prompt Template: Format LLM input
   - LLM: Generate answers

4. ADVANTAGES OF RAG:
   - Accurate answers grounded in documents
   - Reduced hallucination
   - Source attribution
   - Handles private data safely
   - Works with updated documents

5. PRODUCTION CONSIDERATIONS:
   - Scalable vector database
   - Efficient embeddings
   - Multi-user conversation isolation
   - Logging and monitoring
   - Error handling
   - Performance optimization

6. COMMON EXTENSIONS:
   - Multiple document types
   - Hybrid search (text + semantic)
   - Advanced memory strategies
   - Tool integration
   - Custom retrievers
   - Ranking and reranking

7. REAL IMPLEMENTATIONS:
   - LangChain framework
   - LlamaIndex (GPT Index)
   - Haystack
   - Custom solutions
""")

print("\n✅ RAG Agent successfully created!")
print("You've learned the complete flow for building production RAG systems.")