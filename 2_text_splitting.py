"""Exercise 2: Text Splitting and Chunking
Learn how to split documents into manageable chunks for RAG systems."""

print("=" * 70)
print("EXERCISE 2: Text Splitting and Chunking")
print("=" * 70)

# Sample company policy document
document_content = """
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

# Step 1: Demonstrate naive splitting (by line)
print("\n1. NAIVE SPLITTING (BY LINE)")
print("-" * 70)

lines = document_content.split("\n")
print(f"Total lines: {len(lines)}")
print(f"Average line length: {sum(len(line) for line in lines) / len(lines):.1f} characters")

print("\nFirst 5 lines:")
for i, line in enumerate(lines[:5], 1):
    if line.strip():
        print(f"  Line {i}: {line[:60]}...")

# Step 2: Character-based splitting (simple approach)
print("\n\n2. CHARACTER-BASED SPLITTING")
print("-" * 70)

class CharacterTextSplitter:
    """Simple character-based text splitter"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> list:
        """Split text into chunks based on character count"""
        chunks = []
        start = 0

        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size

            # Find next space to avoid cutting words
            if end < len(text):
                # Look for the last space before the chunk_size limit
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start position with overlap
            start = end - self.chunk_overlap

        return chunks

splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
character_chunks = splitter.split_text(document_content)

print(f"Chunk size: 500 characters")
print(f"Chunk overlap: 50 characters")
print(f"Total chunks: {len(character_chunks)}")
print(f"\nChunk sizes:")
for i, chunk in enumerate(character_chunks[:3], 1):
    print(f"  Chunk {i}: {len(chunk)} characters")
    print(f"    Preview: {chunk[:60]}...")

# Step 3: Recursive splitting (smarter approach)
print("\n\n3. RECURSIVE CHARACTER SPLITTING (PREFERRED)")
print("-" * 70)

class RecursiveCharacterTextSplitter:
    """Recursive text splitter that respects document structure"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50,
                 separators: list = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def split_text(self, text: str) -> list:
        """Recursively split text using multiple separators"""
        chunks = []

        def _split(text, separators):
            good_splits = []
            separator = separators[-1]

            for _s in separators:
                if _s == "":
                    break
                if _s in text:
                    separator = _s
                    break

            if separator:
                splits = text.split(separator)
            else:
                splits = list(text)

            good_splits = [s for s in splits if s]

            return good_splits, separator

        def _merge_splits(splits, separator):
            separator_len = len(separator)
            good_splits = []
            current_chunk = []
            total_len = 0

            for s in splits:
                s_len = len(s)
                if total_len + s_len + separator_len > self.chunk_size:
                    if current_chunk:
                        chunk = separator.join(current_chunk)
                        good_splits.append(chunk)
                        current_chunk = []
                        total_len = 0

                current_chunk.append(s)
                total_len += s_len + separator_len

            if current_chunk:
                chunk = separator.join(current_chunk)
                good_splits.append(chunk)

            return good_splits

        splits, separator = _split(text, self.separators)
        good_splits = _merge_splits(splits, separator)

        # Handle overlap
        if self.chunk_overlap > 0:
            final_chunks = []
            for i, chunk in enumerate(good_splits):
                final_chunks.append(chunk)
                if i < len(good_splits) - 1:
                    # Add overlap with next chunk
                    next_chunk = good_splits[i + 1]
                    overlap_chunk = chunk[-self.chunk_overlap:] + " " + next_chunk[:self.chunk_overlap]
                    if overlap_chunk.strip():
                        final_chunks.append(overlap_chunk)
            return final_chunks

        return good_splits

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)
recursive_chunks = recursive_splitter.split_text(document_content)

print(f"Total chunks: {len(recursive_chunks)}")
print(f"\nChunk quality metrics:")
print(f"  Smallest chunk: {min(len(c) for c in recursive_chunks)} characters")
print(f"  Largest chunk: {max(len(c) for c in recursive_chunks)} characters")
print(f"  Average chunk: {sum(len(c) for c in recursive_chunks) / len(recursive_chunks):.0f} characters")

print(f"\nFirst 2 chunks:")
for i, chunk in enumerate(recursive_chunks[:2], 1):
    print(f"\n  Chunk {i}:")
    print(f"    Length: {len(chunk)} characters")
    print(f"    Content: {chunk[:80]}...")

# Step 4: Chunk overlap demonstration
print("\n\n4. UNDERSTANDING CHUNK OVERLAP")
print("-" * 70)

print("""
Chunk overlap is important because:
  1. Prevents losing context at chunk boundaries
  2. Helps semantic search find relevant chunks better
  3. Ensures continuity in multi-chunk responses
  4. Common overlap: 10-20% of chunk size

Example with chunk_size=500 and chunk_overlap=50:
  - Chunk 1: [0 - 500]
  - Chunk 2: [450 - 950]  (last 50 chars overlap)
  - Chunk 3: [900 - 1400] (last 50 chars overlap)
""")

# Step 5: Comparison
print("\n\n5. COMPARISON: CHARACTER vs RECURSIVE SPLITTING")
print("-" * 70)

print(f"Character splitting: {len(character_chunks)} chunks")
print(f"Recursive splitting: {len(recursive_chunks)} chunks")
print(f"\nRecursive splitting is preferred because:")
print(f"  - Respects document structure (paragraphs, sentences)")
print(f"  - Better context preservation")
print(f"  - More semantically meaningful chunks")
print(f"  - Easier for embedding and retrieval")

# Step 6: Summary
print("\n\n" + "=" * 70)
print("KEY CONCEPTS:")
print("=" * 70)
print("""
1. TEXT SPLITTING IMPORTANCE:
   - LLM context windows are limited (typically 4K - 8K tokens)
   - Need to split large documents into chunks
   - Chunks become units for embedding and retrieval

2. SPLITTING STRATEGIES:
   - Character-based: Simple but may cut off mid-sentence
   - Token-based: Respects token limits but needs tokenizer
   - Semantic: Splits on paragraphs/sentences (best for RAG)
   - Recursive: Tries multiple separators hierarchically

3. CHUNK SIZE CONSIDERATIONS:
   - Too small: Loss of context, more retrieval overhead
   - Too large: May not fit in context window, poor retrieval
   - Typical range: 300-1000 characters per chunk
   - Balance between context and token usage

4. CHUNK OVERLAP:
   - Prevents losing context at boundaries
   - Helps semantic search accuracy
   - Typical overlap: 10-20% of chunk size
   - Creates duplication but improves recall

5. BEST PRACTICES:
   - Use recursive splitting with multiple separators
   - Match separators to document structure
   - Test different chunk sizes for your use case
   - Monitor chunk overlap impact on retrieval quality
""")