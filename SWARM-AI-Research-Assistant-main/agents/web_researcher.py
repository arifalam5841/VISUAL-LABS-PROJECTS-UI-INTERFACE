from langchain_ollama import ChatOllama
from ddgs import DDGS


llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


def web_researcher_agent(state):

    subtasks = state["subtasks"]

    all_results = []

    for task in subtasks:

        print(f"\n🔎 Searching for: {task}")

        try:

            with DDGS() as ddgs:

                results = list(
                    ddgs.text(
                        task,
                        max_results=5
                    )
                )

            for result in results:

                all_results.append({
                    "task": task,
                    "title": result.get("title", ""),
                    "url": result.get("href", ""),
                    "snippet": result.get("body", "")
                })

        except Exception as e:

            print(f"Search error: {e}")

    return {
        "web_results": all_results
    }