from langchain_ollama import ChatOllama


llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


def fact_checker_agent(state):

    web_results = state["web_results"]

    academic_results = state["academic_results"]

    print("\n🔍 Fact Checker Agent is analyzing the research...")

    # --------------------------------------------------
    # PREPARE WEB INFORMATION
    # --------------------------------------------------

    web_information = ""

    for item in web_results:

        web_information += f"""
Title: {item['title']}
URL: {item['url']}
Information: {item['snippet']}

"""


    # --------------------------------------------------
    # PREPARE ACADEMIC INFORMATION
    # --------------------------------------------------

    academic_information = ""

    for item in academic_results:

        academic_information += f"""
Paper: {item['title']}
Authors: {', '.join(item['authors'])}
Published: {item['published']}
URL: {item['url']}
Abstract: {item['summary']}

"""


    # --------------------------------------------------
    # FACT CHECKING PROMPT
    # --------------------------------------------------

    prompt = f"""
You are a Fact Checker Agent in an AI Research Assistant.

Your task is to analyze the research information collected
from web sources and academic papers.

WEB SOURCES:

{web_information}


ACADEMIC SOURCES:

{academic_information}


For the most important claims you find:

1. Identify the claim.
2. Determine whether the available sources support it.
3. Look for conflicting evidence.
4. Classify the claim as:

SUPPORTED
CONTRADICTED
UNVERIFIED

For every claim, provide:

Claim:
Status:
Evidence:
Sources:

Do not invent sources or information.
Only use the information provided above.
"""


    response = llm.invoke(prompt)

    verified_claims = response.content

    print("\n✅ Fact checking completed.")

    return {
        "verified_claims": verified_claims
    }