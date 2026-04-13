from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import tool
from langchain.agents import create_agent

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
agent = create_agent(
    model = llm, 
    tools = tools,
    system_prompt = "You are a helpful assistant. Be concise and accurate. Say it if you cannot answer a specific question")

# run it. recursion limit == max iterations
response = agent.invoke(
    {
    "messages": [("human", "How many days until Singapore's next National Day on August 9th? Also, please tell me which tools you used to define this answer")]},
    config={"recursion_limit": 10}
    )

print(response["messages"][-1].content)

