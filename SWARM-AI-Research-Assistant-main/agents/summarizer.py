from langchain_ollama import ChatOllama


llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


def summarizer_agent(state):

    verified_claims = state["verified_claims"]

    print("\n📝 Summarizer Agent is creating the research summary...")

    prompt = f"""
You are the Summarizer Agent in an AI Research Assistant.

Create a clear and informative research summary using
ONLY the verified information provided below.

VERIFIED RESEARCH:

{verified_claims}

Create the summary using the following structure:

1. Introduction
2. Key Findings
3. Current Applications
4. Benefits
5. Limitations
6. Future Scope
7. Conclusion

Rules:

- Do not invent facts.
- Do not create fake statistics.
- Do not add sources that are not provided.
- Clearly distinguish evidence from general conclusions.
- Keep the writing suitable for a college-level research report.
"""

    response = llm.invoke(prompt)

    summary = response.content

    print("\n✅ Research summary completed.")

    return {
        "summary": summary
    }