from langchain_ollama import ChatOllama


llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


def coordinator_agent(state):

    query = state["query"]

    prompt = f"""
You are the Coordinator Agent of an AI Research Assistant.

The user wants to research the following topic:

{query}

Break this research topic into 4 specific research tasks.

The tasks should cover different aspects of the topic, such as:

- Background
- Recent developments
- Research evidence
- Advantages and limitations
- Future scope

Return ONLY the tasks as a numbered list.
"""

    response = llm.invoke(prompt)

    subtasks = response.content.split("\n")

    subtasks = [
        task.strip()
        for task in subtasks
        if task.strip()
    ]

    return {
        "subtasks": subtasks
    }