from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
import os

load_dotenv()

groq = os.getenv("GROQ_API_KEY")


#Websearch Agent
websearch_agent = Agent(
    name="Web search Agent",
    role="Search the web for the information",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True,
)

## Finance Agent
finance_agent = Agent(
    name="Finance AI Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[
        YFinanceTools(stock_price=True, analyst_recommendations= True, stock_fundamentals=True, company_news=True),
    ],
    instructions=["Use table to display the data"],
    show_tool_calls= True,
    markdown=True,
)

multi_ai_agent = Agent(
    team=[websearch_agent, finance_agent],
    model=Groq(id="llama-3.3-70b-versatile"),
    instructions=["Always include sources", "Use table to display the data"],
    show_tool_calls=True,
    markdown=True
)

multi_ai_agent.print_response("Summarize analyst recommendations and share the latest news for NVDA", stream=True)