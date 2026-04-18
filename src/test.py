import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    persist_directory="knowledge_base/chroma_db",
    embedding_function=embeddings
)

test_questions = [
    "What is the student to faculty ratio?",
    "How much does a parking pass cost?",
    "What buildings are on the tour route?",
    "How does a tour guide handle uncomfortable questions?",
    "What is the First Day Complete program?",
]

for question in test_questions:
    print(f"\nQ: {question}")
    print("-" * 50)
    results = vectorstore.similarity_search(question, k=2)
    for r in results:
        print(f"Source: {r.metadata.get('filename', r.metadata.get('source', 'web'))}")
        print(f"Content: {r.page_content[:200]}")
        print()