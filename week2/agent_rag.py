from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain.tools import tool
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import MemorySaver

from langchain_community.tools import DuckDuckGoSearchRun
from datetime import datetime

import os

load_dotenv()

base_dir = os.path.dirname(os.path.abspath(__file__))
docs_path = os.path.join(base_dir, "documents")

# load documents
loader = DirectoryLoader(
    docs_path,
    glob="**/*.txt",
    loader_cls=TextLoader
)
documents = loader.load()
print(f"documents loaded: {len(documents)}")

# Schritt 2: Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)
print(f"Anzahl Chunks: {len(chunks)}")

# load free embedding model to create local FAISS Index
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = FAISS.from_documents(chunks, embeddings)

# save index locally
vectorstore.save_local("faiss_index")
print("Index gespeichert.")

# load the vector store again
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# RAG as tool
@tool
def search_documents(query: str) -> str:
    """Search internal documents and knowledge base. 
    Use this as the first step for any question before searching the web. 
    Only use web_search if no relevant information was found in the documents."""

    results = vectorstore.similarity_search(query, k=3)
    if not results:
        return "No relevant documents found."
    context = "\n\n".join([r.page_content for r in results])
    return f"Found relevant information:\n\n{context}"

# search tool
search_web = DuckDuckGoSearchRun()
search_web.name = "web_search"
search_web.description = (
    "Useful for searching current information on the web. "
    "Use this when you need facts, news, or information not in internal docs."
)

# calculator tool
@tool
def calculator(expression: str) -> str:
    """Evaluates a mathematical expression. Input should be a valid Python math expression."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

@tool
def get_current_date() -> str:
    """Returns the current date. Use it when you need to calculate date differences."""
    return datetime.now().strftime("%Y-%m-%d")

tools = [search_documents, search_web, calculator, get_current_date]

# Agent definition
llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)
memory = MemorySaver()

agent = create_agent(
    model = llm,
    tools = tools,
    system_prompt=(
        "You are a helpful assistant with access to internal documentation "
        "and web search. Always check internal documents first before searching the web. "
        "Be concise and cite which tool you used."
    ),
    checkpointer=memory
)

config = {"configurable": {"thread_id": "rag-session-1"}}

# Test
test_questions = [
    "What is Mosaic AI?",                          # sollte search_documents nutzen
    "What is the current population of Singapore?", # sollte web_search nutzen
    "What did we just discuss?",                    # testet Memory
    "Tell me about Singapore's economy."
]

for question in test_questions:
    print(f"\n>>> {question}")
    response = agent.invoke(
        {"messages": [("human", question)]},
        config=config
    )
    # Alle Messages ausgeben statt nur die letzte
    for msg in response["messages"]:
        print(f"[{msg.type}]: {str(msg.content)}")
    print()