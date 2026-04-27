from duckduckgo_search import DDGS
from skill_registry import skill

@skill(
    name="web_search",
    description="Search the web for current information. Input: query string.",
    parameters={"query": "string"}
)
def web_search(query: str) -> str:
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=3))
    if not results:
        return "No results found."
    formatted = "\n".join([f"{r['title']}: {r['body']}" for r in results])
    return formatted
