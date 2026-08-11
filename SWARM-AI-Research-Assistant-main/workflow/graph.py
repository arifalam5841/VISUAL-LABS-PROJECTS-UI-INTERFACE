from langgraph.graph import StateGraph, START, END

from workflow.state import ResearchState

from agents.coordinator import coordinator_agent
from agents.web_researcher import web_researcher_agent
from agents.academic_researcher import academic_researcher_agent
from agents.fact_checker import fact_checker_agent
from agents.summarizer import summarizer_agent
from agents.report_generator import report_generator_agent


def create_research_graph():

    graph = StateGraph(ResearchState)

    # --------------------------------------------------
    # ADD AGENTS
    # --------------------------------------------------

    graph.add_node(
        "coordinator",
        coordinator_agent
    )

    graph.add_node(
        "web_researcher",
        web_researcher_agent
    )

    graph.add_node(
        "academic_researcher",
        academic_researcher_agent
    )

    graph.add_node(
        "fact_checker",
        fact_checker_agent
    )

    graph.add_node(
        "summarizer",
        summarizer_agent
    )

    graph.add_node(
        "report_generator",
        report_generator_agent
    )

    # --------------------------------------------------
    # START → COORDINATOR
    # --------------------------------------------------

    graph.add_edge(
        START,
        "coordinator"
    )

    # --------------------------------------------------
    # COORDINATOR → RESEARCH AGENTS
    # --------------------------------------------------

    graph.add_edge(
        "coordinator",
        "web_researcher"
    )

    graph.add_edge(
        "coordinator",
        "academic_researcher"
    )

    # --------------------------------------------------
    # RESEARCH AGENTS → FACT CHECKER
    # --------------------------------------------------

    graph.add_edge(
        "web_researcher",
        "fact_checker"
    )

    graph.add_edge(
        "academic_researcher",
        "fact_checker"
    )

    # --------------------------------------------------
    # FACT CHECKER → SUMMARIZER
    # --------------------------------------------------

    graph.add_edge(
        "fact_checker",
        "summarizer"
    )

    # --------------------------------------------------
    # SUMMARIZER → REPORT GENERATOR
    # --------------------------------------------------

    graph.add_edge(
        "summarizer",
        "report_generator"
    )

    # --------------------------------------------------
    # REPORT GENERATOR → END
    # --------------------------------------------------

    graph.add_edge(
        "report_generator",
        END
    )

    return graph.compile()