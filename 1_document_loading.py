"""Exercise 1: Document Loading and Preprocessing
Learn how to load documents from various sources for RAG systems."""

from pathlib import Path
import tempfile

print("=" * 70)
print("EXERCISE 1: Document Loading and Preprocessing")
print("=" * 70)

# Step 1: Create sample documents (simulating loading from files)
print("\n1. CREATING SAMPLE DOCUMENTS")
print("-" * 70)

# In a real RAG system, you might load from:
# - TextLoader: Load .txt files
# - PDFLoader: Load PDF documents
# - WebBaseLoader: Load from URLs
# - etc.

# For now, we'll create sample company policy documents
company_policies = """
INNOVATECH COMPANY POLICIES

Policy 1: Mobile Device Policy
The mobile device policy ensures that all employees maintain productivity and security
while using mobile devices for business purposes. Employees must ensure devices are
password-protected and up-to-date with security patches. Company data should not be
stored on personal devices without proper encryption.

Policy 2: Remote Work Policy
Employees are permitted to work remotely up to 3 days per week, subject to manager approval.
Remote workers must maintain a dedicated workspace and ensure a reliable internet connection.
Video calls should use company-approved platforms only.

Policy 3: Smoking Policy
Smoking is strictly prohibited inside all company buildings and vehicles. Smoking is only
permitted in designated outdoor areas, at least 20 feet away from building entrances.
E-cigarettes and vaping are subject to the same restrictions as traditional cigarettes.

Policy 4: Dress Code
Business casual is the standard dress code at Innovatech. This means slacks or skirts with
a collared shirt or blouse. Denim is not permitted. Exceptions may be made for specific roles
or departments with manager approval.

Policy 5: Internet and Email Policy
Company internet and email are for business purposes only. Employees are responsible for
protecting their login credentials. Inappropriate use of company systems may result in
disciplinary action, up to and including termination.

Policy 6: Vacation and Paid Time Off
Employees receive 20 days of paid vacation annually, plus 10 company holidays. Vacation
requests must be submitted at least 2 weeks in advance. During peak business periods,
vacation may be limited for certain departments.

Policy 7: Code of Conduct
All employees are expected to treat colleagues with respect and professionalism.
Harassment, discrimination, and bullying are strictly prohibited. Violations should be
reported to Human Resources immediately.

Policy 8: Health and Safety
Employees must follow all safety protocols and report hazards immediately. First aid kits
are located on each floor. Emergency procedures are posted in all work areas. Safety
training is mandatory for all new employees.

Policy 9: Confidentiality
All company information, including code, strategies, and business plans, must be kept
confidential. Non-disclosure agreements are required upon hire. Violations may result in
legal action and termination.
"""

print("Sample company policy document created")
print(f"Document length: {len(company_policies)} characters")
print(f"Document preview:\n{company_policies[:200]}...\n")

# Step 2: Save document to a temporary file
print("\n2. SAVING DOCUMENT")
print("-" * 70)

with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
    f.write(company_policies)
    temp_file_path = f.name

print(f"Document saved to: {temp_file_path}")

# Step 3: Load the document using file reading
print("\n3. LOADING DOCUMENT FROM FILE")
print("-" * 70)

with open(temp_file_path, 'r') as f:
    loaded_content = f.read()

print(f"Successfully loaded {len(loaded_content)} characters")
print(f"Content preview:\n{loaded_content[:150]}...\n")

# Step 4: Demonstrate different document formats
print("\n4. SIMULATING DIFFERENT DOCUMENT LOADERS")
print("-" * 70)

# In real LangChain, you would use:
# from langchain_community.document_loaders import TextLoader, PDFLoader, WebBaseLoader

document_sources = {
    "TextLoader": "Load from .txt files",
    "PDFLoader": "Load from PDF documents",
    "WebBaseLoader": "Load from web URLs",
    "CSVLoader": "Load from CSV files",
    "JSONLoader": "Load from JSON files",
    "DirectoryLoader": "Load from entire directories",
}

print("Available document loaders in LangChain:")
for loader_name, description in document_sources.items():
    print(f"  - {loader_name}: {description}")

# Step 5: Parse document into structured format
print("\n\n5. STRUCTURING DOCUMENT DATA")
print("-" * 70)

from typing import List, Dict

class Document:
    """Simple Document class (similar to LangChain's Document)"""
    def __init__(self, page_content: str, metadata: Dict = None):
        self.page_content = page_content
        self.metadata = metadata or {}

# Parse the document into individual policies
policies = []
policy_sections = loaded_content.split("Policy ")

for i, section in enumerate(policy_sections[1:], 1):  # Skip header
    policy_title = section.split("\n")[0].strip()

    doc = Document(
        page_content=f"Policy {section}",
        metadata={
            "source": "company_policies.txt",
            "policy_number": i,
            "policy_title": policy_title,
            "type": "company_policy"
        }
    )
    policies.append(doc)

print(f"Parsed {len(policies)} individual policies")
print(f"\nFirst policy metadata:")
for key, value in policies[0].metadata.items():
    print(f"  {key}: {value}")

print(f"\nFirst policy preview:")
print(f"  {policies[0].page_content[:100]}...\n")

# Step 6: Summary of document loading process
print("\n" + "=" * 70)
print("KEY CONCEPTS:")
print("=" * 70)
print("""
1. DOCUMENT LOADING:
   - Load from various sources (files, URLs, databases, etc.)
   - Preserve metadata about document source and type
   - Handle different formats (text, PDF, web, etc.)

2. DOCUMENT STRUCTURE:
   - page_content: The actual text from the document
   - metadata: Additional info (source, author, date, etc.)
   - Use Document class to standardize format

3. METADATA IMPORTANCE:
   - Track document source for attribution
   - Store document type for filtering
   - Include timestamps and version info
   - Enable source attribution in responses

4. PREPARATION STEPS:
   - Load raw content from source
   - Parse into Document objects
   - Preserve metadata for retrieval
   - Normalize formatting
   - Ready for next step: splitting into chunks

5. REAL-WORLD CONSIDERATIONS:
   - Different file formats need different loaders
   - Large documents may need batch loading
   - Metadata helps trace responses back to sources
   - Consider privacy and sensitive data
""")

# Cleanup
import os
os.unlink(temp_file_path)
print(f"\nCleaned up temporary file: {temp_file_path}")