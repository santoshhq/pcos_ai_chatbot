from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from pinecone import Pinecone
from langchain_core.documents import Document
import re

# =========================
# 🔑 STEP 1: EMBEDDING MODEL
# =========================
client = NVIDIAEmbeddings(
    model="nvidia/nv-embed-v1",
   
)

# =========================
# 🔑 STEP 2: LOAD PDF
# =========================
loader = DirectoryLoader(
    path="heal_her_knowledge_resource",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

docs = loader.lazy_load()

# =========================
# 🔑 STEP 3: MERGE TEXT
# =========================
full_text = "\n".join([doc.page_content for doc in docs])

# =========================
# 🔑 STEP 4: CLEAN TEXT (VERY IMPORTANT)
# =========================
def clean_text(text):
    # 1. Fix broken words across lines
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)

    # 2. Remove repeated headers (VERY IMPORTANT)
    text = re.sub(r'International Evidence-based Guideline.*?\d{4}', '', text)

    # 3. Remove page numbers like "Page 1", "1", etc.
    text = re.sub(r'\n?\s*Page\s*\d+\s*\n?', '\n', text)
    text = re.sub(r'\n\d+\n', '\n', text)

    # 4. Remove references like [1], [23]
    text = re.sub(r'\[\d+\]', '', text)

    # 5. Remove URLs
    text = re.sub(r'http\S+', '', text)

    # 6. Remove copyright / license blocks
    text = re.sub(r'©.*?(?=\n)', '', text)

    # 7. Remove excessive newlines
    text = re.sub(r'\n+', '\n', text)

    # 8. Normalize spaces
    text = re.sub(r'[ \t]+', ' ', text)

    # 9. Fix spacing around punctuation
    text = re.sub(r'\s+([.,])', r'\1', text)

    # 10. Remove weird special characters (keep medical symbols)
    text = re.sub(r'[^\w\s.,()%\-:\n]', '', text)

    return text.strip()

full_text = clean_text(full_text)



pattern = r'(Chapter\s+\w+|[0-9]+\.[0-9]+)'
splits = re.split(pattern, full_text)

documents = []

for i in range(1, len(splits), 2):
    section_title = splits[i]
    content = splits[i + 1]

    documents.append(
        Document(
            page_content=f"{section_title}\n{content.strip()}",
            metadata={
                "section": section_title,
                "source": "PCOS_guideline",
                "type": "medical_guideline"
            }
        )
    )

# =========================
# 🔑 STEP 6: FINAL CHUNKING (BEST SETTINGS)
# =========================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,        # ✅ optimal
    chunk_overlap=150,     # ✅ context retention
    separators=["\n\n", "\n", ".", " "]
)

final_documents = text_splitter.split_documents(documents)

# =========================
# 🔑 STEP 7: PINECONE SETUP
# =========================
pc = Pinecone()
index = pc.Index("healher")

vectorstore = PineconeVectorStore(
    index=index,
    embedding=client
)

# =========================
# 🔑 STEP 8: STORE DOCUMENTS
# =========================
vectorstore.add_documents(final_documents)

print("✅ Data successfully indexed in Pinecone!")
