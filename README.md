# RAG (Retrieval-Augmented Generation) Exercises

Comprehensive hands-on exercises for mastering Retrieval-Augmented Generation (RAG) systems using LangChain, local embeddings, and vector stores.

Based on the Skills Network course: **"Summarize Private Documents Using RAG, LangChain, and LLMs"**

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

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- **langchain** - Framework for chaining components
- **langchain-ollama** - Integration with local Ollama
- **sentence-transformers** - HuggingFace embeddings
- **chromadb** - Vector database for storage

### 2. Setup Ollama (for Exercise 7)

```bash
# Install Ollama from https://ollama.ai
# Then start the server:
ollama serve

# In another terminal, pull the model:
ollama pull qwen2.5:7b
```

### 3. Run Exercises

```bash
python 1_document_loading.py      # Concepts
python 2_text_splitting.py         # Text chunking
python 3_embeddings_vector_store.py # Real embeddings & Chroma
python 4_rag_retrieval_system.py    # RAG pipeline
python 5_rag_prompts.py             # Prompt engineering
python 6_conversation_memory.py     # Memory management
python 7_complete_rag_agent.py      # Full system with Ollama LLM
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
- Production-ready for RAG systems

**Run:**
```bash
python 3_embeddings_vector_store.py
```

**Note:** First run downloads all-MiniLM-L6-v2 model (~100MB)

---

### Exercise 4: Building a RAG Retrieval System
**File:** `4_rag_retrieval_system.py`

Create a production RAG system using HuggingFace embeddings and Chroma.

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

Build a fully-featured RAG agent with local LLM integration.

**Key Concepts:**
- **HuggingFaceEmbeddings** (all-MiniLM-L6-v2) for text encoding
- **Chroma** vector store for semantic search
- **Ollama + Qwen2.5:7b** for local LLM generation
- **ConversationMemory** for multi-turn interactions
- RAG prompt engineering for grounded answers

**Key Takeaways:**
- Complete local RAG system (no API keys, fully offline)
- Ollama provides local LLM inference
- Memory enables conversational RAG
- Fully production-ready architecture

**Prerequisites:**
- Ollama running: `ollama serve`
- Model downloaded: `ollama pull qwen2.5:7b`

**Run:**
```bash
python 7_complete_rag_agent.py
```

**Creates:** `./chroma_rag_complete/` with indexed documents ready for querying

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

## Production Deployment

### Considerations
- **Scalability**: Use managed vector databases (Pinecone, Weaviate)
- **Performance**: Implement caching, approximate search
- **Reliability**: Error handling, fallbacks, monitoring
- **Security**: Encrypt data, control access, audit logs
- **Cost**: Choose efficient embedding models, batch processing
- **Maintenance**: Document updates, version tracking

### Recommended Stack
- **Vector Store**: Chroma (local) / Pinecone (cloud)
- **Embeddings**: sentence-transformers (local) / OpenAI (quality)
- **LLM**: Ollama (local) / Claude/GPT-4 (quality)
- **Framework**: LangChain for orchestration
- **Monitoring**: Logging, metrics, tracing

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

## Performance Checklist

- [ ] Documents are properly loaded and validated
- [ ] Chunks are appropriately sized (300-1000 chars)
- [ ] Embeddings are created and indexed
- [ ] Retrieval returns relevant chunks
- [ ] Prompts prevent hallucination
- [ ] Memory persists conversation history
- [ ] Multi-turn interactions work correctly
- [ ] Response quality meets expectations

## License

Educational - Based on Skills Network course materials

## Notes

- All examples are self-contained and runnable
- No API keys required - everything runs locally
- Simulated LLM responses for demonstration
- Extend with real LLM integration (Ollama, Claude, GPT-4)
- Suitable for learning and production deployment
- MIT License - Free to use and modify

---

**Start with Exercise 1 and work through sequentially for best learning outcomes!**