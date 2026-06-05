"""Exercise 6: Conversation Memory in RAG
Build a stateful conversational RAG agent that maintains chat history."""

print("=" * 70)
print("EXERCISE 6: Conversation Memory and Multi-Turn RAG")
print("=" * 70)

# Step 1: Understanding conversation memory
print("\n1. WHAT IS CONVERSATION MEMORY?")
print("-" * 70)

print("""
Conversation memory allows the AI to:
  - Remember previous exchanges
  - Understand context and references ("it", "that", "those")
  - Maintain coherent multi-turn conversations
  - Build on previous answers

Without memory:
  User: "What is the mobile policy?"
  AI: "The mobile device policy states..."
  User: "What are the main points?"
  AI: [confused, doesn't know what "the policy" refers to]

With memory:
  User: "What is the mobile policy?"
  AI: "The mobile device policy states..."
  User: "What are the main points?"
  AI: "Regarding the mobile policy we just discussed..."
""")

# Step 2: Memory types
print("\n2. TYPES OF MEMORY")
print("-" * 70)

memory_types = {
    "Buffer Memory": {
        "description": "Stores all messages in a buffer",
        "pros": ["Simple", "Full history", "Good for short conversations"],
        "cons": ["Unbounded growth", "Token waste", "Can exceed context window"],
        "best_for": "Short conversations (10-50 turns)",
    },

    "Summary Memory": {
        "description": "Periodically summarizes older messages",
        "pros": ["Bounded size", "Keeps context", "Memory of all turns"],
        "cons": ["Information loss", "Requires summarization"],
        "best_for": "Long conversations (100+ turns)",
    },

    "Entity Memory": {
        "description": "Stores key entities and relationships",
        "pros": ["Focused on important info", "Efficient"],
        "cons": ["Complex to implement", "Requires entity extraction"],
        "best_for": "Complex domain conversations",
    },

    "Vector Memory": {
        "description": "Stores semantic memory of past exchanges",
        "pros": ["Semantic search of history", "Relevant context"],
        "cons": ["Additional complexity", "Embedding overhead"],
        "best_for": "Long conversations with semantic search",
    },

    "Window Memory": {
        "description": "Keeps only last N messages",
        "pros": ["Simple", "Bounded size", "Recent context"],
        "cons": ["Loses old context", "May miss important facts"],
        "best_for": "Very long conversations with limited memory",
    },
}

for mem_type, details in memory_types.items():
    print(f"\n{mem_type}:")
    print(f"  Description: {details['description']}")
    print(f"  Best for: {details['best_for']}")
    print(f"  Pros: {', '.join(details['pros'])}")
    print(f"  Cons: {', '.join(details['cons'])}")

# Step 3: Implementing buffer memory
print("\n\n3. IMPLEMENTING BUFFER MEMORY")
print("-" * 70)

class ConversationBufferMemory:
    """Simple conversation buffer memory"""

    def __init__(self, memory_key: str = "chat_history", human_prefix: str = "User",
                 ai_prefix: str = "Assistant"):
        self.buffer = []
        self.memory_key = memory_key
        self.human_prefix = human_prefix
        self.ai_prefix = ai_prefix

    def add_user_message(self, message: str):
        """Add user message to memory"""
        self.buffer.append({
            "type": "user",
            "content": message,
            "prefix": self.human_prefix
        })

    def add_ai_message(self, message: str):
        """Add AI message to memory"""
        self.buffer.append({
            "type": "ai",
            "content": message,
            "prefix": self.ai_prefix
        })

    def get_buffer(self) -> str:
        """Get formatted buffer as string"""
        if not self.buffer:
            return ""

        lines = []
        for msg in self.buffer:
            lines.append(f"{msg['prefix']}: {msg['content']}")

        return "\n".join(lines)

    def get_messages(self) -> list:
        """Get buffer as list of message dicts"""
        return self.buffer.copy()

    def clear(self):
        """Clear all messages"""
        self.buffer = []

    def __len__(self):
        """Number of messages in buffer"""
        return len(self.buffer)

# Demonstrate buffer memory
print("Creating conversation memory...")
memory = ConversationBufferMemory()

# Simulate multi-turn conversation
exchanges = [
    ("What is the mobile device policy?",
     "The mobile device policy ensures employees maintain security. Devices must be password-protected and have up-to-date security patches."),

    ("What about data storage?",
     "Company data should not be stored on personal devices without proper encryption. Mobile devices must also have antivirus software."),

    ("What if I lose my device?",
     "If you lose your device, you should report it immediately to the IT department. They can help secure or wipe the device remotely."),
]

for i, (user_msg, ai_msg) in enumerate(exchanges, 1):
    print(f"\n\nTurn {i}:")
    print(f"User: {user_msg}")
    print(f"AI: {ai_msg}")

    memory.add_user_message(user_msg)
    memory.add_ai_message(ai_msg)

print(f"\n\nTotal messages in memory: {len(memory)}")
print(f"\nFull conversation history:")
print("-" * 70)
print(memory.get_buffer())

# Step 4: Window memory
print("\n\n4. WINDOW MEMORY (LIMITED HISTORY)")
print("-" * 70)

class ConversationWindowMemory:
    """Keep only last N messages"""

    def __init__(self, k: int = 4, memory_key: str = "chat_history",
                 human_prefix: str = "User", ai_prefix: str = "Assistant"):
        self.buffer = []
        self.k = k
        self.memory_key = memory_key
        self.human_prefix = human_prefix
        self.ai_prefix = ai_prefix

    def add_user_message(self, message: str):
        """Add user message"""
        self.buffer.append({
            "type": "user",
            "content": message,
            "prefix": self.human_prefix
        })
        self._trim_buffer()

    def add_ai_message(self, message: str):
        """Add AI message"""
        self.buffer.append({
            "type": "ai",
            "content": message,
            "prefix": self.ai_prefix
        })
        self._trim_buffer()

    def _trim_buffer(self):
        """Keep only last k messages"""
        if len(self.buffer) > self.k:
            self.buffer = self.buffer[-self.k:]

    def get_buffer(self) -> str:
        """Get formatted buffer"""
        if not self.buffer:
            return ""

        lines = []
        for msg in self.buffer:
            lines.append(f"{msg['prefix']}: {msg['content']}")

        return "\n".join(lines)

# Demonstrate window memory
print("Creating window memory (last 2 message pairs)...")
window_memory = ConversationWindowMemory(k=4)

for user_msg, ai_msg in exchanges:
    window_memory.add_user_message(user_msg)
    window_memory.add_ai_message(ai_msg)

print(f"\nWindow memory (last 4 messages):")
print("-" * 70)
print(window_memory.get_buffer())

print(f"\nNote: Older messages are forgotten to save tokens")

# Step 5: RAG with memory
print("\n\n5. RAG WITH CONVERSATION MEMORY")
print("-" * 70)

class RAGWithMemory:
    """RAG system that includes conversation context"""

    def __init__(self):
        self.memory = ConversationBufferMemory()
        self.context_retriever = self._simple_retriever

    def _simple_retriever(self, query: str) -> str:
        """Simulate retrieving context (normally from vector store)"""
        policy_snippets = {
            "mobile": """Mobile Device Policy:
                - Devices must be password-protected
                - Keep security patches updated
                - Don't store company data without encryption
                - Use VPN on public WiFi
                - Report lost devices immediately""",

            "remote": """Remote Work Policy:
                - Work remotely up to 3 days/week
                - Maintain dedicated workspace
                - Keep reliable internet connection
                - Use approved platforms for video calls
                - Maintain 10 AM - 3 PM core hours""",

            "smoking": """Smoking Policy:
                - Prohibited inside buildings and vehicles
                - Only in designated outdoor areas
                - 20 feet from entrances minimum
                - Same rules for e-cigarettes and vaping""",
        }

        # Simulate simple keyword matching
        for keyword, snippet in policy_snippets.items():
            if keyword.lower() in query.lower():
                return snippet

        return "No specific policy found."

    def answer_question(self, user_query: str, use_memory: bool = True) -> str:
        """Answer question with optional conversation context"""
        # Retrieve context
        context = self.context_retriever(user_query)

        # Build prompt
        prompt_parts = []

        if use_memory and len(self.memory) > 0:
            prompt_parts.append("Previous conversation:")
            prompt_parts.append(self.memory.get_buffer())
            prompt_parts.append("")

        prompt_parts.append("Context from policies:")
        prompt_parts.append(context)
        prompt_parts.append("")
        prompt_parts.append(f"Question: {user_query}")
        prompt_parts.append("Answer:")

        prompt = "\n".join(prompt_parts)

        # Simulate LLM response
        answer = f"Based on the policy information, I can tell you that..."

        # Store in memory
        self.memory.add_user_message(user_query)
        self.memory.add_ai_message(answer)

        return answer

# Demonstrate RAG with memory
print("Simulating RAG with conversation memory...")

rag = RAGWithMemory()

test_queries = [
    "What is the mobile device policy?",
    "Can I store company data on my personal phone?",
    "What if I lose it?",
]

for query in test_queries:
    print(f"\n\nQuery: {query}")
    answer = rag.answer_question(query, use_memory=True)
    print(f"Answer: {answer}")

print(f"\n\nConversation memory after all queries:")
print("-" * 70)
print(rag.memory.get_buffer())

# Step 6: Token management
print("\n\n6. TOKEN MANAGEMENT IN CONVERSATION MEMORY")
print("-" * 70)

print("""
Managing tokens with growing conversation history:

Problem: Each turn adds tokens, eventually exceeding context window

Solutions:

1. BUFFER MEMORY:
   Pro: Simple, complete history
   Con: Unbounded growth
   When: Short conversations (<50 turns)

2. WINDOW MEMORY:
   Pro: Bounded size, recent context
   Con: Loses old information
   When: Long conversations (100+ turns)

3. SUMMARY MEMORY:
   Pro: Keeps overview of all exchanges
   Con: Information loss
   When: Very long conversations

4. HYBRID APPROACH:
   - Keep last N turns in buffer (recent)
   - Summarize older turns
   - Reference summary when needed

Token Budget Example:
  - Context window: 4096 tokens
  - System message: 200 tokens
  - Retrieved context: 800 tokens
  - Available for history: 3096 tokens
  - With 100 tokens/turn: 30 turns max
""")

# Step 7: Best practices
print("\n\n" + "=" * 70)
print("KEY CONCEPTS:")
print("=" * 70)
print("""
1. CONVERSATION MEMORY TYPES:
   - Buffer: Full history, simple, unbounded
   - Window: Recent N messages, bounded
   - Summary: Summarized history, efficient
   - Entity: Key facts and relationships
   - Vector: Semantic memory with search

2. IMPLEMENTATION PATTERNS:
   - Add user message to memory
   - Get response from LLM/RAG
   - Add AI message to memory
   - Include memory in next prompt

3. CONTEXT MANAGEMENT:
   - Include previous exchanges for context
   - Reference previous answers
   - Build on prior knowledge
   - Maintain conversation coherence

4. TOKEN OPTIMIZATION:
   - Monitor total token count
   - Trim or summarize when needed
   - Use window memory for long conversations
   - Reserve tokens for context retrieval

5. MULTI-TURN INTERACTIONS:
   - User: "What is the mobile policy?"
   - AI: [Answer with context]
   - User: "What about data encryption?"
   - AI: [Can reference previous about mobile policy]

6. PRACTICAL CONSIDERATIONS:
   - Clear memory between sessions
   - Persist memory for user continuity
   - Measure conversation quality
   - Test different memory sizes
   - Monitor token usage

7. COMMON PATTERNS:
   - Chat applications: Use buffer memory
   - Support bots: Use window memory
   - Analysis tools: Use summary memory
   - Enterprise: Use persistent memory
""")