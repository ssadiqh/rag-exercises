# RAG (Retrieval-Augmented Generation) Exercises

**Beginner-friendly local RAG learning exercises** — Step-by-step hands-on exercises for learning Retrieval-Augmented Generation (RAG) systems using LangChain, HuggingFace embeddings, Chroma vector store, and local Ollama LLM.

Based on the Skills Network course: **"Summarize Private Documents Using RAG, LangChain, and LLMs"**

⚠️ **Learning & Teaching Focus:** These exercises teach RAG concepts with runnable code. Not designed for production deployment without additional hardening (error handling, scaling, monitoring, security).

## What is RAG?

**Retrieval-Augmented Generation (RAG)** is a technique that combines document retrieval with language model generation to create AI systems that:
- Answer questions about private documents
- Reduce hallucination by grounding responses in actual content
- Handle documents introduced after model training cutoff
- Provide traceable, source-attributed answers
- Maintain confidentiality by processing documents locally

## Stack: Full Local Setup (Option 2 - Recommended)

| Component | Choice | Details |
|-----------|--------|---------|
| **Embeddings** | HuggingFace (all-MiniLM-L6-v2) | 384 dimensions, lightweight, fast |
| **Vector Store** | Chroma (Local SQLite) | Persistent local database, no server |
| **LLM** | Ollama + Qwen2.5:7b | Local, no API keys, fully offline |
| **Framework** | LangChain | Orchestration and chain management |

## Prerequisites

### Required
- **Python 3.8+**
- **pip** package manager

### For Exercise 7 (Complete RAG Agent)
- **Ollama** installed from https://ollama.ai
- **Qwen2.5:7b** model: `ollama pull qwen2.5:7b`
- **Ollama running** in background: `ollama serve`

### Verify Ollama Setup
```bash
# Test Ollama is running
curl http://localhost:11434/api/generate -d '{"model":"qwen2.5:7b","prompt":"hello"}'
```

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/ssadiqh/rag-exercises.git
cd rag-exercises
```

### 2. Create Virtual Environment

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- **langchain** - RAG framework and orchestration
- **langchain-ollama** - Local LLM integration
- **sentence-transformers** - HuggingFace embeddings (all-MiniLM-L6-v2)
- **chromadb** - Local vector database (SQLite)

### 4. Setup Ollama (Optional, Required for Exercise 7)

```bash
# Install Ollama from https://ollama.ai
# Then start the server (Terminal 1):
ollama serve

# In another terminal (Terminal 2), pull the model:
ollama pull qwen2.5:7b

# Verify it's working:
curl http://localhost:11434/api/generate -d '{"model":"qwen2.5:7b","prompt":"test"}'
```

### 5. Run Exercises

**Exercises 1-6** (work without Ollama):
```bash
python 1_document_loading.py      # Document loading concepts
python 2_text_splitting.py         # Text chunking strategies
python 3_embeddings_vector_store.py # HuggingFace embeddings + Chroma
python 4_rag_retrieval_system.py    # RAG pipeline with real components
python 5_rag_prompts.py             # Prompt engineering for RAG
python 6_conversation_memory.py     # Conversation memory types
```

**Exercise 7** (requires Ollama running):
```bash
python 7_complete_rag_agent.py      # Complete system with local LLM
```

**No API keys required — everything runs locally!**

## Exercises Overview

### Exercise 1: Document Loading and Preprocessing
**File:** `1_document_loading.py`

Learn how to load documents from various sources and prepare them for RAG.

**Key Concepts:**
- Simulating different document loaders (TextLoader, PDFLoader, WebBaseLoader)
- Creating structured Document objects
- Preserving metadata for source attribution
- Handling different file formats

**Key Takeaways:**
- Documents are loaded from various sources
- Metadata tracks important information
- Structured format enables downstream processing

**Run:**
```bash
python 1_document_loading.py
```

---

### Exercise 2: Text Splitting and Chunking
**File:** `2_text_splitting.py`

Master the art of splitting large documents into manageable chunks.

**Key Concepts:**
- Character-based text splitting
- Recursive character splitting with multiple separators
- Chunk size and overlap tuning
- Semantic boundary preservation

**Key Takeaways:**
- Document length exceeds LLM context windows
- Chunks need semantic boundaries
- Overlap prevents losing context
- Recursive splitting respects document structure

**Run:**
```bash
python 2_text_splitting.py
```

---

### Exercise 3: Embeddings and Vector Storage
**File:** `3_embeddings_vector_store.py`

Learn to use real HuggingFace embeddings and Chroma vector store.

**Key Concepts:**
- HuggingFaceEmbeddings with all-MiniLM-L6-v2 (384 dimensions)
- Cosine similarity calculation
- **Chroma** vector database (local SQLite-based)
- Persistent storage with automatic indexing
- Semantic search instead of keyword matching

**Key Takeaways:**
- Real embeddings from sentence-transformers
- Chroma provides local, persistent storage
- Semantic search uses vector similarity
- Demonstrates vector storage for RAG systems

**Run:**
```bash
python 3_embeddings_vector_store.py
```

**Note:** First run downloads all-MiniLM-L6-v2 model (~100MB)

---

### Exercise 4: Building a RAG Retrieval System
**File:** `4_rag_retrieval_system.py`

Build a working RAG system using HuggingFace embeddings and Chroma.

**Key Concepts:**
- RecursiveCharacterTextSplitter with semantic boundaries
- HuggingFaceEmbeddings (all-MiniLM-L6-v2) for document chunks
- Chroma vector store with metadata preservation
- Similarity search for retrieval (top-k results)
- Context assembly and source attribution

**Key Takeaways:**
- End-to-end RAG pipeline with real components
- Chroma persists data to disk automatically
- Vector similarity enables semantic retrieval
- Metadata tracks document source and chunk info

**Run:**
```bash
python 4_rag_retrieval_system.py
```

**Output:** Creates `./chroma_rag_data/` directory with indexed documents

---

### Exercise 5: RAG Prompts and Chains
**File:** `5_rag_prompts.py`

Master prompt engineering for RAG systems.

**Key Concepts:**
- Prompt templates with placeholders
- RAG-specific instructions (constraints)
- Prompt engineering strategies (zero-shot, few-shot, chain-of-thought)
- System messages vs. user messages
- Output format specification

**Key Takeaways:**
- Good prompts improve answer quality
- Constraints prevent hallucination
- Output format guides structure
- System messages set AI behavior

**Run:**
```bash
python 5_rag_prompts.py
```

---

### Exercise 6: Conversation Memory in RAG
**File:** `6_conversation_memory.py`

Build stateful conversational RAG agents with memory.

**Key Concepts:**
- Conversation memory types (buffer, window, summary, entity)
- Multi-turn interaction support
- Token management in long conversations
- Memory isolation between sessions
- Maintaining conversation coherence

**Key Takeaways:**
- Memory enables multi-turn conversations
- Different strategies for different use cases
- Token budgets require memory optimization
- Conversation history provides context

**Run:**
```bash
python 6_conversation_memory.py
```

---

### Exercise 7: Complete RAG Agent with Ollama LLM
**File:** `7_complete_rag_agent.py`

Build a working RAG agent that brings together all components with local LLM integration.

**Key Concepts:**
- **HuggingFaceEmbeddings** (all-MiniLM-L6-v2) for text encoding
- **Chroma** vector store for semantic search
- **Ollama + Qwen2.5:7b** for local LLM generation
- **ConversationMemory** for multi-turn interactions
- RAG prompt engineering for grounded answers

**Key Takeaways:**
- Complete local RAG system (no API keys, fully offline)
- Ollama provides accessible local LLM inference
- Memory enables multi-turn conversational RAG
- Shows the full data flow: Query → Embed → Retrieve → Generate → Remember

**Prerequisites:**
- Ollama running: `ollama serve`
- Model downloaded: `ollama pull qwen2.5:7b`

**Run:**
```bash
python 7_complete_rag_agent.py
```

**Creates:** `./chroma_rag_complete/` with indexed documents for interactive querying

---

## RAG Architecture Overview

```
User Query
    ↓
1. INDEXING (Offline, one-time)
    - Load documents
    - Split into chunks
    - Generate embeddings
    - Store in vector database
    ↓
2. RETRIEVAL (At query time)
    - Embed user query
    - Search vector database
    - Retrieve relevant chunks
    ↓
3. CONTEXT ASSEMBLY
    - Format retrieved chunks
    - Add source metadata
    - Assemble prompt
    ↓
4. GENERATION
    - Include conversation history
    - Pass to LLM
    - Generate grounded answer
    ↓
Answer with Source Attribution
```

## Key Concepts Summary

### 1. **Documents and Chunks**
- Documents are loaded from various sources
- Split into semantic chunks (300-1000 chars)
- Metadata preserved for attribution
- Chunks become RAG units

### 2. **Embeddings**
- Convert text to high-dimensional vectors
- Capture semantic meaning
- Enable similarity search
- Typical dimensions: 384-3072

### 3. **Vector Stores**
- Index embeddings for fast search
- Return top-k similar chunks
- Store metadata with chunks
- Enable semantic search

### 4. **Prompts**
- Templates with placeholders
- Control LLM behavior
- Specify output format
- Prevent hallucination

### 5. **Memory**
- Buffer: Complete history
- Window: Recent N turns
- Summary: Condensed history
- Token optimization required

### 6. **Complete Pipeline**
- Seamless data flow
- From documents to answers
- Source attribution
- Multi-turn support

## Embedding Models - Option 2 Stack

| Model | Dimensions | Speed | Quality | Status |
|-------|-----------|-------|---------|--------|
| **all-MiniLM-L6-v2** | 384 | ⚡⚡⚡ Fast | Good | ✓ **Using** |
| all-mpnet-base-v2 | 768 | ⚡⚡ Medium | Excellent | Easy upgrade |
| all-distilroberta-v1 | 768 | ⚡⚡⚡ Fast | High | Alternative |
| OpenAI text-embedding-3-small | 1536 | ⚡ Slow | Very High | Paid API |

**Current Choice:** `all-MiniLM-L6-v2` balances speed (1000 docs/sec) and quality for learning

## Common RAG Patterns

### Simple Q&A
```python
# Query → Retrieve → Generate Answer
results = vector_store.search(query, k=3)
context = format_context(results)
answer = llm(context, query)
```

### Conversational RAG
```python
# Include chat history in context
history = memory.get_buffer()
results = vector_store.search(query, k=3)
answer = llm(history, context, query)
memory.add(query, answer)
```

### Hybrid Search
```python
# Combine keyword + semantic search
keyword_results = keyword_search(query)
semantic_results = semantic_search(query)
combined = merge_results(keyword_results, semantic_results)
```

## Performance Optimization

### Retrieval
- Use approximate nearest neighbor search (HNSW, IVF)
- Batch process embeddings
- Cache embeddings to avoid recomputation
- Tune chunk size for your use case

### Memory Management
- Use window memory for long conversations
- Summarize old messages periodically
- Implement token budgets
- Clear old conversations

### LLM Calls
- Batch multiple queries when possible
- Cache repeated prompts
- Use smaller models when appropriate
- Monitor token usage

## Extending RAG Systems

### 1. Multiple Document Types
```python
# Support PDF, web, database, etc.
documents = load_multiple_sources([
    pdf_path, web_url, database_query
])
```

### 2. Hybrid Search
```python
# Combine keyword and semantic search
results = keyword_search(query) + semantic_search(query)
```

### 3. Re-ranking
```python
# Re-rank retrieved documents by relevance
results = retrieve(query, k=10)
reranked = rerank(results, query, k=3)
```

### 4. Tool Integration
```python
# Use RAG as tool for agents
agent = Agent(tools=[rag_tool, search_tool, calculator])
```

### 5. Streaming
```python
# Stream LLM output for better UX
for token in llm.stream(prompt):
    print(token, end="", flush=True)
```

## From Learning to Production

These exercises teach RAG concepts. **Moving to production requires:**

### Additional Components Needed
- **Error Handling**: Graceful failures, retry logic, input validation
- **Monitoring**: Logging, metrics collection, performance tracking
- **Scalability**: Distributed vector stores (Pinecone, Weaviate, Milvus)
- **Security**: Data encryption, access control, audit trails, prompt injection prevention
- **Performance**: Caching, batch processing, approximate nearest neighbor search
- **Quality**: Retrieval metrics, answer evaluation, user feedback loops
- **Maintenance**: Document versioning, model updates, performance monitoring

### Suggested Production Stack
- **Vector Store**: Pinecone (cloud) or Weaviate (self-hosted) for scalability
- **Embeddings**: sentence-transformers (local) OR OpenAI (managed, higher quality)
- **LLM**: Claude/GPT-4 (quality) OR self-hosted Ollama (privacy-focused)
- **Framework**: LangChain or LlamaIndex for orchestration
- **Monitoring**: ELK stack, Prometheus, or managed logging (Datadog, New Relic)
- **Testing**: Unit tests, integration tests, retrieval quality metrics

## Troubleshooting

### Low Retrieval Quality
- Increase chunk overlap
- Reduce chunk size
- Use better embedding model
- Add more documents
- Re-rank results

### Hallucination Issues
- Add stronger constraints to prompt
- Reduce LLM temperature
- Use grounded generation techniques
- Verify context coverage

### Token Limit Issues
- Reduce chunk overlap
- Use window memory
- Implement summarization
- Optimize prompt length

### Slow Retrieval
- Use approximate search (HNSW)
- Reduce k (number of results)
- Use smaller embedding model
- Implement caching

## Next Steps

After completing these exercises:

1. **Integrate Real LLM**: Connect to Ollama, Claude, or GPT-4
2. **Use Real Vector DB**: Implement Chroma or Pinecone backend
3. **Build Application**: Create web UI or API service
4. **Production Deployment**: Set up scalable infrastructure
5. **Advanced Techniques**: Implement re-ranking, hybrid search, agents
6. **Evaluation**: Measure retrieval quality, answer accuracy
7. **Monitoring**: Track performance, user feedback, costs

## Further Reading

- [LangChain Documentation](https://python.langchain.com/)
- [RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
- [Vector Store Comparison](https://python.langchain.com/docs/integrations/vectorstores/)
- [Embedding Models](https://huggingface.co/models?pipeline_tag=sentence-similarity)
- [Retrieval-Augmented Generation Paper](https://arxiv.org/abs/2005.11401)
- [LLM Best Practices](https://platform.openai.com/docs/guides/production-best-practices)

## Exercises Difficulty Progression

```
Beginner → Intermediate → Advanced
    ↓
1. Document Loading (Concepts)
    ↓
2. Text Splitting (Practical)
    ↓
3. Embeddings (Theory + Practice)
    ↓
4. Retrieval System (Integration)
    ↓
5. Prompts (Optimization)
    ↓
6. Memory (State Management)
    ↓
7. Complete Agent (Production-Ready)
```

## Learning Tips

1. **Run exercises sequentially** - Each builds on previous concepts
2. **Modify and experiment** - Change chunk sizes, temperatures, k values
3. **Read comments carefully** - Code comments explain the "why"
4. **Try different queries** - Test with questions of various difficulty
5. **Monitor output** - Watch how changes affect retrieval and generation
6. **Take notes** - Document your learnings and insights
7. **Build projects** - Apply concepts to real problems

## Common Mistakes to Avoid

1. ❌ Too large chunk sizes (>2000 chars) - Loses semantic boundaries
2. ❌ No chunk overlap - Loses context at boundaries
3. ❌ Poor prompt design - Allows hallucination
4. ❌ No conversation memory - Loses context in multi-turn
5. ❌ Ignoring metadata - Can't attribute sources
6. ❌ Using wrong embedding model - Poor semantic search
7. ❌ Not testing with real data - May fail in production

## Learning Checklist

After completing all 7 exercises, you should understand:

- [ ] Document loading from various sources
- [ ] Text splitting strategies and trade-offs
- [ ] How embeddings work (semantic vectors)
- [ ] Vector similarity search for retrieval
- [ ] RAG prompt engineering best practices
- [ ] Conversation memory types and management
- [ ] How local LLMs integrate with RAG
- [ ] The complete RAG data flow (load → split → embed → store → retrieve → generate)

## Important Limitations to Know

These exercises do **NOT** cover:

- ❌ Error handling and edge cases
- ❌ Production scalability (single-machine only)
- ❌ Security hardening (encryption, access control)
- ❌ Monitoring and observability
- ❌ Prompt injection prevention
- ❌ Retrieval quality metrics
- ❌ User feedback loops
- ❌ Cost optimization for APIs
- ❌ Advanced RAG patterns (hybrid search, reranking, query routing)

## What This Repository IS and IS NOT

### ✅ This Repository IS:
- **A learning resource** for understanding RAG architecture and components
- **Beginner-friendly** with step-by-step progression
- **Runnable code** that you can modify and experiment with
- **Locally-focused** (no cloud dependencies or API keys for Exercises 1-6)
- **Conceptually accurate** about how RAG systems work
- **A teaching workbook** for getting hands-on experience

### ❌ This Repository IS NOT:
- **Production-ready** (missing error handling, monitoring, security)
- **Suitable for large-scale data** (Chroma is local SQLite, not distributed)
- **Enterprise-grade** (no authentication, audit trails, or compliance features)
- **Optimized** (local embeddings, single-machine vector store)
- **Covering advanced techniques** (hybrid search, reranking, query routing)
- **A framework** (it's exercises, not a reusable library)

### Use Cases:
- ✅ Learning RAG fundamentals (weeks 1-2 of a course)
- ✅ Understanding component interactions
- ✅ Building local prototypes
- ✅ Teaching RAG to others
- ❌ Deploying to production
- ❌ Processing large document collections
- ❌ Multi-user systems

---

## License

Educational - Based on Skills Network course materials. MIT License - Free to use and modify.

## Notes

- All examples are self-contained and runnable
- No API keys required for Exercises 1-6 (everything runs locally)
- Exercise 7 requires local Ollama server (also free)
- Designed for learning, not production
- Extend with real deployments (Pinecone, OpenAI, managed infrastructure)
- Feedback welcome via GitHub issues

---

**Start with Exercise 1 and work through sequentially for best learning outcomes!**