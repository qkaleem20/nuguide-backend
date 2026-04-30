import os
import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever

load_dotenv()


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Loading Vector Store

def load_vectorstore():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    client = chromadb.CloudClient(
        tenant=os.getenv("CHROMA_TENANT"),
        database=os.getenv("CHROMA_DATABASE"),
        api_key=os.getenv("CHROMA_API_KEY"),
    )
    return Chroma(
        collection_name="nuguide_kb",
        embedding_function=embeddings,
        client=client
    )

# Faomatting Documets

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Building Chain

def load_chunks():
    """Fetch all chunks from Chroma Cloud for BM25 index."""
    from langchain_core.documents import Document

    client = chromadb.CloudClient(
        tenant=os.getenv("CHROMA_TENANT"),
        database=os.getenv("CHROMA_DATABASE"),
        api_key=os.getenv("CHROMA_API_KEY"),
    )
    collection = client.get_collection("nuguide_kb")

    chunks = []
    offset = 0
    batch_size = 250  # Chroma Cloud limits 300 per request

    while True:
        results = collection.get(
            include=["documents", "metadatas"],
            limit=batch_size,
            offset=offset,
        )
        if not results["documents"]:
            break
        for doc_text, metadata in zip(results["documents"], results["metadatas"]):
            chunks.append(Document(page_content=doc_text, metadata=metadata))
        if len(results["documents"]) < batch_size:
            break
        offset += batch_size

    print(f"  Loaded {len(chunks)} chunks from Chroma Cloud for BM25 index.")
    return chunks

def build_chain(vectorstore):
    llm = ChatOpenAI(
        model="gpt-5.4-mini-2026-03-17",
        temperature=0,
    )

    # Semantic Search
    semantic_retriever = vectorstore.as_retriever(
        search_type = "similarity",
        search_kwargs = {"k": 4}
    )

    # BM25 Keyword search
    chunks = load_chunks()
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 8

    def hybrid_retrieve(question):
        semantic_docs = semantic_retriever.invoke(question)
        bm25_docs = bm25_retriever.invoke(question)

        seen = set()
        combined = []
        for doc in semantic_docs + bm25_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                combined.append(doc)
        
        def source_priority(doc):
            source = doc.metadata.get("filename", "")
            if source == "academic_cheat_sheet.pdf":
                return 0
            elif source in ["academics", "key_facts", "tour_route", "useful_links", "college_bridges", "nu_clubs", "operations"]:
                return 1
            elif doc.metadata.get("source_type") == "handbook":
                return 2
            else:
                return 3
        combined.sort(key=source_priority)
        return combined[:12]

    # Step 1: Reformulate follow-up questions using chat history
    # so the vector search gets a proper standalone query
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant for Niagara University tour guides. "
         "Given the chat history and the latest question, reformulate the question "
         "to be fully standalone so it can be understood without the chat history. "
         "Do NOT answer it. Just return the reformulated question as plain text."),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ])
    # Step 2: Main answer prompt
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a personal assistant built specifically for Niagara University tour guides.

Your primary purpose is to help tour guides — especially new ones or those returning from breaks — quickly look up accurate university information without having to search \
through handbooks or training materials. Think of yourself as a knowledgeable teammate they can ask anything before, during, or after a tour.

You also help guides prepare confident answers for questions from prospective students and families.

STRICT RULES:
- Answer ONLY using the information provided in the context below.
- If the context does not contain enough information to answer, say exactly:
  "I don't have that in my materials. Please reach out to Sara, the Co-Directors, or call the admissions office at 716-286-8700."
- Do NOT use outside knowledge or make anything up.
- Be direct and concise — guides need fast, reliable, clear answers, not long explanations.
- For questions about specific academic programs, majors, minors, or scholarships, always prioritize information from the academic cheat sheet (academic_cheat_sheet.pdf) over website sources. 
  The cheat sheet is the most reliable and up-to-date source maintained by the admissions office.
  If the cheat sheet has relevant information, use it as your primary source and supplement with website content only if needed.
- If a question involves an exact number (tuition, parking cost, class size, etc.), quote it exactly as it appears in the materials. Never estimate.
- If a question about academic programs or majors is vague (for example, "tell me about business" or "what programs does NU have"), ask ONE focused follow-up question before answering.
  For example: "Are you looking for undergraduate programs, graduate programs, or a specific major?" 
  Do not ask follow-up questions for factual lookups like parking, hours, or statistics.
- When a guide asks where they can find more information or a full list of something, provide the relevant link from your materials if available.
- If a guide asks about graduate programs, graduate admissions, or anything related to graduate studies, do not attempt to answer. Instead say exactly: 
  "Graduate programs are handled separately. Please contact the Graduate Studies office directly at 716-286-7360."
- After answering a question about a specific academic program, major, or campus facility, if there is additional relevant information in your context that the guide might find useful on tour, proactively offer it. 
  For example: "I can also tell you about the study abroad options for this program" or "Would you like the quick tour-guide version of the key facts for this major?" Keep the offer to one short sentence at the end of your answer. 
  Do not do this for simple factual lookups like parking costs or phone numbers.
- Always be professional, positive, and accurate about Niagara University.

Context:
{context}"""),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ])

    # Chain assembly
    #
    # How this works step by step:
    # 1. Reformulate the question using chat history
    # 2. Retrieve relevant chunks from vector store
    # 3. Format chunks into a single string
    # 4. Pass context + history + question to LLM
    # 5. Parse output as plain string

    def retrieve_with_context(inputs):
        question = inputs["question"]
        chat_history = inputs["chat_history"]

        # If there's chat history, reformulate the question first
        if chat_history:
            reformulation_chain = contextualize_prompt | llm | StrOutputParser()
            question = reformulation_chain.invoke({
                "question": question,
                "chat_history": chat_history
            })

        # Retrieve relevant docs using the (possibly reformulated) question
        docs = hybrid_retrieve(question)
        return {
            "context": format_docs(docs),
            "source_documents": docs,
            "question": inputs["question"],  # original question for the answer
            "chat_history": chat_history
        }

    def generate_answer(inputs):
        chain = answer_prompt | llm | StrOutputParser()
        answer = chain.invoke({
            "context": inputs["context"],
            "question": inputs["question"],
            "chat_history": inputs["chat_history"]
        })
        return {
            "answer": answer,
            "source_documents": inputs["source_documents"]
        }

    # Combine into one callable
    def full_chain(inputs):
        retrieved = retrieve_with_context(inputs)
        return generate_answer(retrieved)

    return full_chain

# Ask a question 

def ask(chain, question, chat_history):
    result = chain({
        "question": question,
        "chat_history": chat_history
    })

    answer = result["answer"]
    sources = result["source_documents"]

    # Detect if the LLM gave the fallback response
    fallback_phrases = [
        "I don't have that in my materials",
        "Graduate programs are handled separately"
    ]
    is_unanswered = any(phrase.lower() in answer.lower() for phrase in fallback_phrases)

    seen = set()
    unique_sources = []
    for doc in sources:
        name = doc.metadata.get("filename") or doc.metadata.get("source", "website")
        if name not in seen:
            seen.add(name)
            unique_sources.append(name)

    return answer, unique_sources, is_unanswered, len(sources)

# Main Loop

if __name__ == "__main__":
    print("Loading knowledge base...")
    vectorstore = load_vectorstore()
    chain = build_chain(vectorstore)

    print("\nNU Tour Guide Assistant is ready.")
    print("Type 'quit' to exit.\n")
    print("-" * 50)

    chat_history = []

    while True:
        question = input("\nYou: ").strip()

        if not question:
            continue
        if question.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        answer, sources, is_unanswered, sources_found = ask(chain, question, chat_history)

        # Update history for follow-up questions
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=answer))

        print(f"\nAssistant: {answer}")
        print(f"\nSources: {', '.join(sources)}")
        print("-" * 50)