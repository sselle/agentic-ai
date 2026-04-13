from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

from datetime import datetime

load_dotenv()

search_tool = DuckDuckGoSearchRun()
search_tool.name = "web_search"
search_tool.description = (
    "Useful for searching current information on the web. "
    "Use this when you need facts, news, or information you don't know"
)

@tool
def calculator(expression: str) -> str:
    """Evaluates a mathematical expression. Input should be a valid Python math expression."""
    try:
        return(str(eval(expression)))
    except Exception as e:
        return f"Error: {e}"
    
@tool
def get_current_date() -> str:
    """Returns the current date and time. Use it when you need to calculate date differences"""
    return datetime.now().strftime("%Y-%m-%d")

tools = [search_tool, calculator, get_current_date]
llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)

# Memory via checkpointer
memory = MemorySaver() 

agent = create_agent(
    model = llm,
    tools = tools,
    system_prompt="You are a helpful assistant. Be concise and accurate.",
    checkpointer=memory
)

# Wichtig: thread_id identifiziert die Konversation
config = {"configurable": {"thread_id": "conversation-1"}}

# Multi-turn Test
questions = [
    "What is the GDP of Singapore?",
    "How does that compare to Germany?",
    "What is the ratio between the two numbers?"
]

# iterate over questions
for question in questions:
    print(f"\n>>> {question}")
    response = agent.invoke(
        {"messages": [("human", question)]},
        config=config  # gleiche config = gleiche Konversation
    )
    print(f"Agent: {response['messages'][-1].content}")

config = {"configurable": {"thread_id": "conversation-2"}}

print("\n--- Interaktiver Modus (quit zum Beenden) ---\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    response = agent.invoke(
        {"messages": [("human", user_input)]},
        config=config
    )
    print(f"Agent: {response['messages'][-1].content}\n")