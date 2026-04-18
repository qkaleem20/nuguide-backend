import os
import re
import pickle
from datetime import date 
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PDFPlumberLoader,
    WebBaseLoader,
    UnstructuredWordDocumentLoader,
    TextLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import chromadb

load_dotenv()
os.environ.setdefault("USER_AGENT", "NUGuide/1.0")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define Sources

DOCUMENT_FILES = [
    (os.path.join(PROJECT_ROOT, "knowledge_base/documents/student_room_handbook.docx"), "handbook"),
    (os.path.join(PROJECT_ROOT, "knowledge_base/documents/spring_2026_training.pdf"), "handbook"),
    (os.path.join(PROJECT_ROOT, "knowledge_base/documents/academic_cheat_sheet.pdf"), "handbook"),
    (os.path.join(PROJECT_ROOT, "knowledge_base/documents/nu_clubs.pdf"), "nu_clubs"),
    (os.path.join(PROJECT_ROOT, "knowledge_base/documents/campus_tour_guide_handbook.docx"), "handbook"),
    (os.path.join(PROJECT_ROOT, "knowledge_base/documents/tour_route.txt"), "tour_route"),
    (os.path.join(PROJECT_ROOT, "knowledge_base/documents/general_facts.txt"), "key_facts"),
    (os.path.join(PROJECT_ROOT, "knowledge_base/documents/useful_links.txt"), "useful_links"),
    (os.path.join(PROJECT_ROOT, "knowledge_base/documents/college_bridges.txt"), "college_bridges")
]

WEBSITE_URLS = [
    "https://www.niagara.edu/tuition-aid/scholarships-grants/",
    "https://www.niagara.edu/tuition-aid/financial-aid/",
    "https://www.niagara.edu/academics/",
    "https://www.niagara.edu/programs/",
    "https://www.niagara.edu/colleges/college-of-arts-and-sciences/",
    "https://www.niagara.edu/colleges/college-of-education/",
    "https://www.niagara.edu/colleges/college-of-hospitality-sport-and-tourism-management/",
    "https://www.niagara.edu/colleges/college-of-nursing/",
    "https://www.niagara.edu/colleges/holzschuh-college-of-business-administration/",
    "https://www.niagara.edu/academics/academic-and-career-exploration-program/",
    "https://www.niagara.edu/academics/honors-program/",
    "https://www.niagara.edu/admissions/international-admissions/",
    "https://www.niagara.edu/admissions/canadian-students/",
    "https://www.niagara.edu/current-students/student-life/clubs-organizations/"
]

COLEECTION_NAME = "nuguide_kb"

def clean_web_content(text):
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'(Skip to content|Toggle navigation|Search|Menu|Back to top)', '', text)
    text = re.sub(r'^\s*https?://\S+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r' {3,}', ' ', text)
    return text.strip()

# Load Documents

def load_documents():
    all_docs = []

    # Local Load files
    for filepath, source_type in DOCUMENT_FILES:
        print(f"Loading: {filepath}")
        if filepath.endswith(".pdf"):
            loader = PDFPlumberLoader(filepath)
        elif filepath.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(filepath, mode="single")
        elif filepath.endswith(".txt"):
            loader = TextLoader(filepath)
        else:
            continue

        docs = loader.load()
        for doc in docs:
            # This will confirm if your 17-page Word doc is actually being read
            print(f"    - Content length: {len(doc.page_content)} characters")
            doc.metadata["source_type"] = source_type
            doc.metadata["filename"] = os.path.basename(filepath)
            doc.metadata["ingested_date"] = str(date.today())

        all_docs.extend(docs)
        print(f"  → {len(docs)} document objects loaded")
    
    print("\nLoading Website pages...")
    loaded_count = 0
    for url in WEBSITE_URLS:
        try:
            loader = WebBaseLoader(
                [url],
                header_template={
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
    )
            loader.requests_kwargs = {'timeout': 15}
            web_docs = loader.load()
            for doc in web_docs:
                doc.page_content = clean_web_content(doc.page_content)
                doc.metadata["source_type"] = "website"
                doc.metadata["source"] = url
                doc.metadata["ingested_date"] = str(date.today())
            all_docs.extend(web_docs)
            loaded_count += 1
            print(f"  ✓ {url}")
        except Exception as e:
            print(f"  ! Error Loading Website: {url} — {str(e)[:60]}")
    print(f"\n  → {loaded_count}/{len(WEBSITE_URLS)} web pages loaded")

    return all_docs

# Splitting into chunks

def chunk_documents(docs):
    pdf_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1500,
        chunk_overlap = 150,
        separators = ["\n\n\n", "\n\n", "\n", ".", " "]
    )
    web_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 80,
        separators = ["\n\n", "\n", ".", " "]
    )

    txt_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 512,
        chunk_overlap = 50,
        separators = ["\n\n", "\n", ".", " "]
    )

    all_chunks = []
    for doc in docs:
        source_type = doc.metadata.get("source_type", "")
        filename = doc.metadata.get("filename", "")

        if source_type == "website":
            chunks = web_splitter.split_documents([doc])
        elif filename.endswith(".txt"):
            chunks = txt_splitter.split_documents([doc])
        else:
            chunks = pdf_splitter.split_documents([doc])

        all_chunks.extend(chunks)

    # Filter out near-empty chunks
    all_chunks = [c for c in all_chunks if len(c.page_content.strip()) > 50]

    print(f"\nTotal chunks created: {len(all_chunks)}")
    return all_chunks

def save_chunks(chunks):
    chunks_path = "knowledge_base/chunks.pkl"
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Chunks saved to {chunks_path}")

# Embedding and storing

def build_vector_store(chunks):
    print("\n Connecting to Chroma Cloud...")

    client = chromadb.CloudClient(
        tenant=os.getenv("CHROMA_TENANT"),
        database=os.getenv("CHROMA_DATABASE"),
        api_key=os.getenv("CHROMA_API_KEY"),
    )

    try:
        client.delete_collection("nuguide_kb")
        print (f" Cleared existing '{"nuguide_kb"}' collection")
    except Exception:
        print(f" No existing collection to clear. Creating fresh.")

    print("Generating embeddings and uploading to Chroma Cloud...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    BATCH_SIZE = 200
    vectorstore = None

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f" Uploading Batch {batch_num}/{total_batches} ({len(batch)} chunks)...")

        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                collection_name="nuguide_kb",
                client=client,
            )
        else:
            vectorstore.add_documents(batch)

    print(f"Vector store built on Chroma cloud. {len(chunks)} chunks indexed.")
    return vectorstore

# Run

if __name__ == "__main__":
    docs = load_documents()
    chunks = chunk_documents(docs)
    save_chunks(chunks)
    build_vector_store(chunks)
    print("\nIngestion complete. Knowledge Base is ready")