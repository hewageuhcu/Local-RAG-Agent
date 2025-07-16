from langchain_community.tools.tavily_search import TavilySearchResults

# Web Search Tool
web_search_tool = TavilySearchResults(k=3)