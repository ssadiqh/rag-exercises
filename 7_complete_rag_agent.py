"""Exercise 7: Complete RAG Agent with Ollama LLM
Build a full-featured RAG agent combining HuggingFace embeddings, Chroma, and local Ollama LLM."""

import tempfile
import os

print("=" * 70)
print("EXERCISE 7: Complete RAG Agent with Local LLM")
print("=" * 70)

# Step 0: Prerequisites check
print("\n0. CHECKING PREREQUISITES")
print("-" * 70)

print("""
Required for this exercise:
  1. Ollama running locally (localhost:11434)
  2. Qwen2.5:7b model downloaded

Setup Ollama:
  - Install from: https://ollama.ai
  - Run: ollama serve
  - Pull model: ollama pull qwen2.5:7b
  - Test: curl http://localhost:11434/api/generate -d '{"model":"qwen2.5:7b","prompt":"test"}'

Verify Ollama is running before proceeding...
""")

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

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
with open(doc_path, 'r') as f:
    chunks = splitter.split_text(f.read())

print(f"Split into {len(chunks)} chunks")

# Step 3: Create documents
print("\n3. CREATING DOCUMENT OBJECTS")
print("-" * 70)

from langchain_core.documents import Document

documents = []
for i, chunk in enumerate(chunks):
    doc = Document(
        page_content=chunk,
        metadata={"source": "policies.txt", "chunk_id": i}
    )
    documents.append(doc)

print(f"Created {len(documents)} documents")

# Step 4: Embeddings and Vector Store
print("\n4. SETTING UP EMBEDDINGS AND VECTOR STORE")
print("-" * 70)

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_chroma import Chroma

    print("Loading HuggingFaceEmbeddings (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("Creating Chroma vector store...")
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="./chroma_rag_complete"
    )
    print(f"✓ Vector store ready with {len(documents)} documents")

except ImportError as e:
    print(f"⚠️ Error: {e}")
    print("Install with: pip install sentence-transformers chromadb")
    vector_store = None
    embeddings = None

# Step 5: Initialize Ollama LLM
print("\n5. INITIALIZING OLLAMA LLM")
print("-" * 70)

try:
    from langchain_ollama import OllamaLLM

    print("Connecting to Ollama (localhost:11434)...")
    print("Model: qwen2.5:7b")

    llm = OllamaLLM(
        model="qwen2.5:7b",
        temperature=0.3,
        base_url="http://localhost:11434"
    )

    # Test connection
    test_response = llm.invoke("Hello")
    print(f"✓ Ollama connection successful")
    print(f"✓ Test response received ({len(test_response)} chars)")

except ImportError as e:
    print(f"⚠️ Error: {e}")
    print("Install with: pip install langchain-ollama")
    print("Also ensure Ollama is running: ollama serve")
    llm = None

except Exception as e:
    print(f"⚠️ Connection error: {e}")
    print("Make sure Ollama is running: ollama serve")
    print("And Qwen model is available: ollama pull qwen2.5:7b")
    llm = None

# Step 6: Conversation memory
print("\n6. SETTING UP CONVERSATION MEMORY")
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
        lines = [f"{m['role'].upper()}: {m['content']}" for m in self.messages[-4:]]  # Last 4 for context
        return "\n".join(lines)

    def clear(self):
        self.messages = []

memory = ConversationMemory()
print("✓ Conversation memory initialized")

# Step 7: RAG Prompt template
print("\n7. CREATING RAG PROMPT TEMPLATE")
print("-" * 70)

RAG_PROMPT = """You are a helpful assistant answering questions about Innovatech company policies.

INSTRUCTIONS:
1. Use ONLY the provided context to answer questions
2. If the context doesn't contain the answer, say "I don't have information about that"
3. Be concise but complete
4. Reference which policy you're citing when relevant

{chat_history}
CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

print("✓ RAG prompt template ready")

# Step 8: Complete RAG Agent
print("\n8. COMPLETE RAG AGENT")
print("-" * 70)

class RAGAgent:
    def __init__(self, vector_store, llm, memory, embeddings):
        self.vector_store = vector_store
        self.llm = llm
        self.memory = memory
        self.embeddings = embeddings
        self.turn_count = 0

    def answer(self, query: str, verbose: bool = True) -> str:
        if not self.vector_store or not self.llm:
            print("⚠️ Vector store or LLM not available")
            return ""

        self.turn_count += 1

        if verbose:
            print(f"\n{'='*60}")
            print(f"Turn {self.turn_count}")
            print(f"User: {query}")
            print('-'*60)

        # Retrieve context
        results = self.vector_store.similarity_search(query, k=3)
        context = "\n\n".join([r.page_content for r in results])

        # Get chat history
        chat_history = self.memory.get_history()
        if chat_history:
            chat_history = f"PREVIOUS CONVERSATION:\n{chat_history}\n\n"

        # Format prompt
        prompt = RAG_PROMPT.format(
            chat_history=chat_history,
            context=context,
            question=query
        )

        # Get LLM response
        try:
            answer = self.llm.invoke(prompt)
        except Exception as e:
            print(f"⚠️ LLM error: {e}")
            answer = "I'm having trouble generating a response. Please ensure Ollama is running."

        if verbose:
            print(f"Assistant: {answer[:200]}...")
            print(f"Sources: {', '.join([r.metadata.get('source', 'unknown') for r in results])}")

        # Store in memory
        self.memory.add_user_message(query)
        self.memory.add_assistant_message(answer)

        return answer

# Initialize agent
if vector_store and llm:
    agent = RAGAgent(vector_store, llm, memory, embeddings)
    print("✓ RAG Agent initialized")

    # Step 9: Demo conversation
    print("\n9. MULTI-TURN CONVERSATION DEMO")
    print("=" * 70)

    demo_queries = [
        "What is the mobile device policy?",
        "Can I store company data on my phone?",
        "Can I work from home?",
    ]

    for query in demo_queries:
        try:
            agent.answer(query)
        except KeyboardInterrupt:
            print("\n\nConversation interrupted")
            break
        except Exception as e:
            print(f"Error: {e}")

    # Step 10: Show conversation memory
    print("\n10. CONVERSATION MEMORY")
    print("=" * 70)
    print(memory.get_history())

else:
    print("\n⚠️ Cannot initialize agent - vector store or LLM not available")

# Step 11: Complete workflow
print("\n\n" + "=" * 70)
print("COMPLETE RAG SYSTEM ARCHITECTURE")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────┐
│           COMPLETE LOCAL RAG SYSTEM STACK                   │
└─────────────────────────────────────────────────────────────┘

COMPONENTS:
  1. Documents
     └─ Text files, PDFs, web content

  2. Text Splitter
     └─ RecursiveCharacterTextSplitter
        (600 chars, 100 overlap)

  3. Embeddings
     └─ HuggingFaceEmbeddings (all-MiniLM-L6-v2)
        (384-dimensional vectors)

  4. Vector Database
     └─ Chroma (Local SQLite)
        (./chroma_rag_complete/)

  5. Language Model
     └─ Ollama + Qwen2.5:7b
        (localhost:11434)

  6. Memory
     └─ ConversationBufferMemory
        (Last 4 turns for context)

WORKFLOW:

  USER QUERY
       ↓
  Embed query (all-MiniLM-L6-v2)
       ↓
  Search Chroma vector store (top-3)
       ↓
  Assemble context with metadata
       ↓
  Include conversation history
       ↓
  Format RAG prompt
       ↓
  Send to Ollama LLM
       ↓
  Generate grounded answer
       ↓
  Store in conversation memory
       ↓
  Return to user

BENEFITS:
  ✓ Fully local (no API keys, no cloud)
  ✓ Offline-capable (after initial setup)
  ✓ Accurate (grounded in documents)
  ✓ Traceable (source attribution)
  ✓ Memory-aware (multi-turn conversations)
  ✓ Fast (local processing)
  ✓ Cost-effective (open source)
""")

# Step 12: Summary and next steps
print("\n" + "=" * 70)
print("KEY CONCEPTS:")
print("=" * 70)
print("""
1. COMPLETE LOCAL RAG:
   ✓ HuggingFaceEmbeddings: all-MiniLM-L6-v2 (384 dims)
   ✓ Chroma: Local SQLite vector database
   ✓ Ollama: Local LLM server (Qwen2.5:7b)
   ✓ Memory: Conversation history management

2. PRODUCTION-READY ARCHITECTURE:
   ✓ No external dependencies
   ✓ No API keys required
   ✓ Runs completely offline
   ✓ Easy to scale to cloud (Pinecone, OpenAI)

3. MULTI-TURN CONVERSATIONS:
   ✓ Maintains conversation history
   ✓ Provides context for follow-up questions
   ✓ Efficient token management (last 4 turns)

4. DOCUMENT GROUNDING:
   ✓ Answers based on actual documents
   ✓ Reduced hallucination
   ✓ Source attribution
   ✓ Works with proprietary documents

5. EXTENSIBILITY:
   - Switch embeddings: all-mpnet-base-v2 (better quality)
   - Switch LLM: Mistral, Llama2, etc.
   - Upgrade vector store: Pinecone, Weaviate
   - Add hybrid search: Keyword + semantic

6. OPTIMIZATION:
   - Adjust chunk size (300-1000 chars)
   - Tune similarity threshold (k=1-5)
   - Adjust LLM temperature (0.1-0.9)
   - Implement re-ranking for better results
""")

# Cleanup
os.unlink(doc_path)

print(f"\n✅ EXERCISE 7 COMPLETE!")
print(f"Vector store saved to: ./chroma_rag_complete/")
print(f"\nNow you have a fully functional local RAG system!")
print(f"Try modifying the demo queries and testing with your own documents.")