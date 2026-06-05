"""Exercise 5: RAG Prompts and Chain
Learn how to create effective prompts for RAG systems using LangChain."""

print("=" * 70)
print("EXERCISE 5: RAG Prompts and Chains")
print("=" * 70)

# Step 1: Understanding prompt templates
print("\n1. PROMPT TEMPLATES FOR RAG")
print("-" * 70)

print("""
Prompt templates are strings with placeholders that get filled in at runtime:

Key Variables in RAG:
  - {context}: The retrieved documents/chunks
  - {question}: The user's query
  - {chat_history}: Previous conversation turns

Good RAG Prompts Should:
  1. Instruct the model to use only provided context
  2. Tell model what to do if context doesn't answer question
  3. Ask for source attribution
  4. Format output clearly
  5. Set appropriate tone/style
""")

# Step 2: Basic prompt template
print("\n2. BASIC RAG PROMPT")
print("-" * 70)

basic_prompt = """Answer the question based on the provided context.

Context:
{context}

Question: {question}

Answer:"""

print("Basic Prompt Template:")
print(basic_prompt)

# Step 3: Intermediate prompt template
print("\n\n3. INTERMEDIATE RAG PROMPT (WITH CONSTRAINTS)")
print("-" * 70)

intermediate_prompt = """You are a helpful assistant answering questions about company policies.
Use ONLY the provided context to answer the question.

If the context does not contain enough information to answer the question,
respond with "I don't have enough information to answer that question."

Do NOT make up or assume information not in the context.

Context:
{context}

Question: {question}

Answer:"""

print("Intermediate Prompt Template:")
print(intermediate_prompt)

# Step 4: Advanced prompt template
print("\n\n4. ADVANCED RAG PROMPT (STRUCTURED)")
print("-" * 70)

advanced_prompt = """You are an expert assistant answering questions about company policies at Innovatech.

INSTRUCTIONS:
1. Answer ONLY using the provided context
2. If context is insufficient, clearly state: "I cannot find this information in the company policies"
3. Be concise but comprehensive
4. Use bullet points for lists
5. Cite which policy section you're referencing

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

print("Advanced Prompt Template:")
print(advanced_prompt)

# Step 5: Simulate prompt template rendering
print("\n\n5. SIMULATING PROMPT TEMPLATE RENDERING")
print("-" * 70)

class PromptTemplate:
    """Simple prompt template class"""

    def __init__(self, template: str, input_variables: list):
        self.template = template
        self.input_variables = input_variables

    def format(self, **kwargs) -> str:
        """Fill in template with provided values"""
        result = self.template

        for var in self.input_variables:
            if var not in kwargs:
                raise ValueError(f"Missing required variable: {var}")

            result = result.replace(f"{{{var}}}", str(kwargs[var]))

        return result

# Create prompt template
prompt_template = PromptTemplate(
    template=intermediate_prompt,
    input_variables=["context", "question"]
)

# Sample context (simulating retrieved chunks)
context = """
Mobile Device Policy - Section 1:
The mobile device policy ensures that all employees maintain productivity and security.
Employees must ensure devices are password-protected and up-to-date with security patches.
Company data should not be stored on personal devices without proper encryption.

Mobile Device Policy - Section 2:
Mobile devices used for company purposes must have antivirus software installed.
VPN must be used when connecting to company networks from public WiFi.
Lost or stolen devices should be reported immediately to IT department.
"""

question = "What should I do if I lose my company mobile device?"

# Format prompt
formatted_prompt = prompt_template.format(context=context, question=question)

print("Formatted Prompt:")
print("=" * 70)
print(formatted_prompt)
print("=" * 70)

# Step 6: Different prompt strategies
print("\n\n6. PROMPT ENGINEERING STRATEGIES FOR RAG")
print("-" * 70)

strategies = {
    "Zero-Shot RAG": """
    Use context directly without examples.
    Best for: General questions

    Template:
    Based on this context: {context}
    Answer: {question}
    """,

    "Few-Shot RAG": """
    Include examples in the prompt.
    Best for: Specific format requirements

    Example:
    Q: What is the dress code?
    A: Business casual dress code policy states...

    Now answer: {question}
    """,

    "Chain-of-Thought RAG": """
    Ask model to reason step-by-step.
    Best for: Complex questions

    Let's think step by step about {question}
    First, find relevant sections in context...
    """,

    "Structured Output RAG": """
    Ask for specific output format.
    Best for: Data extraction

    Answer in JSON format:
    {{
        "answer": "...",
        "confidence": "high/medium/low",
        "source": "policy name"
    }}
    """,
}

for strategy, description in strategies.items():
    print(f"\n{strategy}:")
    print(description)

# Step 7: System vs User prompts
print("\n7. SYSTEM MESSAGES AND CHAT PROMPTS")
print("-" * 70)

system_message = """You are an expert AI assistant specializing in company policies at Innovatech.

Your responsibilities:
1. Answer questions about company policies accurately
2. Use only provided context - never make up policies
3. Be helpful and professional
4. Clearly indicate when you don't have information
5. Provide specific policy references when applicable

Always prioritize accuracy over being helpful."""

user_query_template = """Question: {question}

Please answer based on the following company policy information:

{context}

Provide a clear, accurate answer."""

print("System Message (sets AI behavior):")
print(system_message)

print("\n\nUser Query Template (specific question):")
print(user_query_template)

# Step 8: Chat history in prompts
print("\n\n8. CONVERSATION MEMORY IN RAG")
print("-" * 70)

multi_turn_prompt = """Previous conversation:
{chat_history}

Context from documents:
{context}

Current question: {question}

Answer:"""

chat_history_example = """
User: What is the mobile device policy?
Assistant: The mobile device policy requires devices to be password-protected and up-to-date with security patches.

User: What about lost devices?
Assistant: Lost or stolen devices should be reported immediately to the IT department.

User: How quickly should I report it?
Assistant: You should report it immediately - no specific time frame is specified.
"""

print("Multi-turn Conversation Template:")
print(multi_turn_prompt)

print("\n\nExample Chat History:")
print(chat_history_example)

formatted_multi_turn = multi_turn_prompt.replace(
    "{chat_history}", chat_history_example
).replace(
    "{question}", "Can someone else recover my data?"
).replace(
    "{context}", context
)

print("\nFormatted Conversation Prompt:")
print("=" * 70)
print(formatted_multi_turn)
print("=" * 70)

# Step 9: Output formatting
print("\n\n9. OUTPUT FORMATTING INSTRUCTIONS")
print("-" * 70)

output_formats = {
    "Plain Text": "Just provide the answer in clear prose.",

    "Bullet Points": """
    Format your answer as a bullet list:
    - Point 1
    - Point 2
    - Point 3
    """,

    "Numbered List": """
    Use numbered list format:
    1. First point
    2. Second point
    3. Third point
    """,

    "Markdown": """
    Use markdown formatting:
    # Title
    **bold text**
    - list items
    """,

    "JSON": """
    Return as JSON:
    {
        "answer": "...",
        "confidence": "...",
        "sources": [...]
    }
    """,

    "Structured": """
    Follow this structure:
    ANSWER: [brief answer]
    EXPLANATION: [detailed explanation]
    SOURCES: [which policies referenced]
    CONFIDENCE: [high/medium/low]
    """,
}

for format_name, instructions in output_formats.items():
    print(f"\n{format_name}:")
    print(instructions)

# Step 10: Best practices
print("\n\n" + "=" * 70)
print("KEY CONCEPTS:")
print("=" * 70)
print("""
1. PROMPT TEMPLATES:
   - Reusable templates with placeholders
   - Fill in variables at runtime
   - Ensure consistency across queries

2. RAG-SPECIFIC INSTRUCTIONS:
   - Tell model to use only context
   - Specify what to do when context is insufficient
   - Request source attribution
   - Set appropriate constraints

3. PROMPT ENGINEERING:
   - Zero-shot: Simple, direct prompts
   - Few-shot: Include examples
   - Chain-of-thought: Encourage reasoning
   - Structured: Request specific format

4. CONVERSATION CONTEXT:
   - Include chat history for multi-turn
   - Maintain conversation coherence
   - Reference previous exchanges
   - Manage token usage

5. OUTPUT CONTROL:
   - Specify format (text, JSON, markdown)
   - Use structured instructions
   - Request confidence levels
   - Ask for source attribution

6. TESTING & ITERATION:
   - Test prompts with various questions
   - Evaluate answer quality
   - Refine based on results
   - Document working prompts

7. COMMON MISTAKES:
   - Forgetting to include context
   - Allowing model to hallucinate
   - Unclear constraints
   - No output format specification
   - Ignoring conversation history
""")