import os
import re
import pickle
from datetime import date
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PDFPlumberLoader,
    WebBaseLoader,
    UnstructuredWordDocumentLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import chromadb

load_dotenv()
os.environ.setdefault("USER_AGENT", "NUGuide/1.0")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCUMENTS_DIR = os.path.join(PROJECT_ROOT, "knowledge_base", "documents")
SYNCED_DIR = os.path.join(DOCUMENTS_DIR, "synced")
LOCAL_DIR = os.path.join(DOCUMENTS_DIR, "local")

COLLECTION_NAME = "nuguide_kb"
DEFAULT_SOURCE_TYPE = "general"

SOURCE_TYPE_MAP = {
    # --- synced from Drive (new consolidated set) ---
    "01_Key_Facts_and_Stats.docx": "key_facts",
    "02_Campus_Buildings.docx": "campus_buildings",
    "03_Student_Services.docx": "student_services",
    "04_Local_Area_and_Transportation.docx": "local_area",
    "05_Campus_Directory_and_Contacts.docx": "contacts",
    "06_Clubs_and_Student_Life.docx": "nu_clubs",
    "07_Tour_Route.docx": "tour_route",
    "08_Tour_Guide_Operations.docx": "operations",
    "10_College_Quick_Reference.docx": "academics",
    "academic_cheat_sheet.docx": "handbook",  # pinned name; see chatbot.py
    # --- local-only (not in Drive) ---
    "nu_clubs.pdf": "nu_clubs",
    "spring_2026_training.pdf": "handbook",
    "student_room_handbook.docx": "handbook",
    "campus_tour_guide_handbook.docx": "handbook",
}

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
    "https://www.niagara.edu/current-students/student-life/clubs-organizations/",
    "https://www.niagara.edu/about/nu-history/",
    "https://www.niagara.edu/future-students/admitted-students/",
    "https://www.niagara.edu/athletics/",
    "https://www.niagara.edu/current-students/student-life/kiernan-recreation-center/",
    "https://catalog.niagara.edu/undergraduate/student-affairs/residence-life/",
    "https://catalog.niagara.edu/undergraduate/curriculum/foundation-courses/",
    "https://www.niagara.edu/visit/tour-guides/",
    "https://www.niagara.edu/admissions/meet-the-admissions-team/",
]


def clean_web_content(text):
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(Skip to content|Toggle navigation|Search|Menu|Back to top)", "", text)
    text = re.sub(r"^\s*https?://\S+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r" {3,}", " ", text)
    return text.strip()


def _discover_files():
    """Return [(filepath, source_type), ...] from synced/ and local/."""
    found = []
    for folder in (SYNCED_DIR, LOCAL_DIR):
        if not os.path.isdir(folder):
            continue
        for fname in sorted(os.listdir(folder)):
            if fname.startswith("."):
                continue
            if not fname.lower().endswith((".pdf", ".docx", ".txt")):
                continue
            source_type = SOURCE_TYPE_MAP.get(fname, DEFAULT_SOURCE_TYPE)
            found.append((os.path.join(folder, fname), source_type))
    return found


def load_documents():
    all_docs = []

    document_files = _discover_files()
    if not document_files:
        print(
            "WARNING: no files found in synced/ or local/. "
            "Run drive_sync.py first, or check the folders exist."
        )

    for filepath, source_type in document_files:
        print(f"Loading: {filepath}  [{source_type}]")
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
            print(f"    - Content length: {len(doc.page_content)} characters")
            doc.metadata["source_type"] = source_type
            doc.metadata["filename"] = os.path.basename(filepath)
            doc.metadata["ingested_date"] = str(date.today())

        all_docs.extend(docs)
        print(f"  -> {len(docs)} document objects loaded")

    print("\nLoading website pages...")
    loaded_count = 0
    for url in WEBSITE_URLS:
        try:
            loader = WebBaseLoader(
                [url],
                header_template={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                },
            )
            loader.requests_kwargs = {"timeout": 15}
            web_docs = loader.load()
            for doc in web_docs:
                doc.page_content = clean_web_content(doc.page_content)
                doc.metadata["source_type"] = "website"
                doc.metadata["source"] = url
                doc.metadata["ingested_date"] = str(date.today())
            all_docs.extend(web_docs)
            loaded_count += 1
            print(f"  ok {url}")
        except Exception as e:
            print(f"  ! Error loading website: {url} - {str(e)[:60]}")
    print(f"\n  -> {loaded_count}/{len(WEBSITE_URLS)} web pages loaded")

    return all_docs


def chunk_documents(docs):
    pdf_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=150, separators=["\n\n\n", "\n\n", "\n", ".", " "]
    )
    web_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=80, separators=["\n\n", "\n", ".", " "]
    )
    txt_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512, chunk_overlap=50, separators=["\n\n", "\n", ".", " "]
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

    all_chunks = [c for c in all_chunks if len(c.page_content.strip()) > 50]
    print(f"\nTotal chunks created: {len(all_chunks)}")
    return all_chunks


def save_chunks(chunks):
    chunks_path = os.path.join(PROJECT_ROOT, "knowledge_base", "chunks.pkl")
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Chunks saved to {chunks_path}")


def build_vector_store(chunks):
    print("\nConnecting to Chroma Cloud...")
    client = chromadb.CloudClient(
        tenant=os.getenv("CHROMA_TENANT"),
        database=os.getenv("CHROMA_DATABASE"),
        api_key=os.getenv("CHROMA_API_KEY"),
    )

    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Cleared existing '{COLLECTION_NAME}' collection")
    except Exception:
        print("No existing collection to clear. Creating fresh.")

    print("Generating embeddings and uploading to Chroma Cloud...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    BATCH_SIZE = 200
    vectorstore = None
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"Uploading batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                collection_name=COLLECTION_NAME,
                client=client,
            )
        else:
            vectorstore.add_documents(batch)

    print(f"Vector store built. {len(chunks)} chunks indexed.")
    return vectorstore


def run_ingest():
    """Full pipeline. Callable from the API. Returns a summary dict."""
    docs = load_documents()
    chunks = chunk_documents(docs)
    save_chunks(chunks)
    build_vector_store(chunks)
    summary = {
        "documents_loaded": len(docs),
        "chunks_indexed": len(chunks),
    }
    print("\nIngestion complete. Knowledge base is ready.")
    return summary


if __name__ == "__main__":
    run_ingest()