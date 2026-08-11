import streamlit as st
from pathlib import Path

from workflow.graph import create_research_graph


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Swarm AI Research Assistant",
    page_icon="🤖",
    layout="wide"
)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("🐝 About the Swarm")

    st.write(
        """
        This application uses multiple specialized
        AI agents to perform research.
        """
    )

    st.divider()

    st.subheader("🤖 AI Agents")

    st.write("🧠 Coordinator")
    st.write("🔎 Web Researcher")
    st.write("📚 Academic Researcher")
    st.write("🔍 Fact Checker")
    st.write("📝 Summarizer")
    st.write("📄 Report Generator")

    st.divider()

    st.caption("LLM: Ollama + Llama 3.2")
    st.caption("Framework: LangGraph")


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🤖 Swarm AI Research Assistant")

st.write(
    """
    Enter a research topic and let a team of specialized
    AI agents investigate the topic, analyze sources,
    verify important claims, summarize the findings,
    and generate a downloadable research report.
    """
)


# --------------------------------------------------
# RESEARCH INPUT
# --------------------------------------------------

st.subheader("🔍 Research Topic")

query = st.text_area(
    "Enter your research topic:",
    placeholder=(
        "Example: Artificial Intelligence "
        "in Healthcare"
    ),
    height=100
)


# --------------------------------------------------
# START RESEARCH
# --------------------------------------------------

start_research = st.button(
    "🚀 Start Research",
    type="primary"
)


if start_research:

    # --------------------------------------------------
    # VALIDATE INPUT
    # --------------------------------------------------

    if not query.strip():

        st.warning(
            "Please enter a research topic."
        )

        st.stop()


    # --------------------------------------------------
    # SWARM ACTIVITY
    # --------------------------------------------------

    st.subheader("🐝 Swarm Activity")

    progress = st.progress(0)

    status_text = st.empty()


    # --------------------------------------------------
    # AGENT STATUS PLACEHOLDERS
    # --------------------------------------------------

    coordinator_status = st.empty()
    web_status = st.empty()
    academic_status = st.empty()
    fact_status = st.empty()
    summary_status = st.empty()
    report_status = st.empty()


    # --------------------------------------------------
    # INITIAL AGENT STATUS
    # --------------------------------------------------

    coordinator_status.info(
        "🧠 Coordinator Agent: Waiting..."
    )

    web_status.info(
        "🔎 Web Researcher: Waiting..."
    )

    academic_status.info(
        "📚 Academic Researcher: Waiting..."
    )

    fact_status.info(
        "🔍 Fact Checker: Waiting..."
    )

    summary_status.info(
        "📝 Summarizer: Waiting..."
    )

    report_status.info(
        "📄 Report Generator: Waiting..."
    )


    # --------------------------------------------------
    # CREATE GRAPH
    # --------------------------------------------------

    graph = create_research_graph()


    # --------------------------------------------------
    # INITIAL STATE
    # --------------------------------------------------

    initial_state = {

        "query": query,

        "subtasks": [],

        "web_results": [],

        "academic_results": [],

        "findings": [],

        "verified_claims": [],

        "summary": "",

        "final_report": ""
    }


    try:

        # --------------------------------------------------
        # AGENT ORDER
        # --------------------------------------------------

        agent_order = [
            "coordinator",
            "web_researcher",
            "academic_researcher",
            "fact_checker",
            "summarizer",
            "report_generator"
        ]


        total_agents = len(
            agent_order
        )


        completed_nodes = set()


        # --------------------------------------------------
        # STORE FINAL STATE
        # --------------------------------------------------

        result = initial_state.copy()


        # --------------------------------------------------
        # START SWARM
        # --------------------------------------------------

        status_text.info(
            "🐝 Swarm is starting..."
        )


        # --------------------------------------------------
        # STREAM LANGGRAPH
        # --------------------------------------------------

        for update in graph.stream(
            initial_state,
            stream_mode="updates"
        ):

            # --------------------------------------------------
            # PROCESS NODE UPDATES
            # --------------------------------------------------

            for node_name, node_update in update.items():

                # ----------------------------------------------
                # UPDATE RESULT
                # ----------------------------------------------

                if isinstance(
                    node_update,
                    dict
                ):

                    result.update(
                        node_update
                    )


                # ----------------------------------------------
                # PREVENT DUPLICATE PROGRESS
                # ----------------------------------------------

                if node_name in completed_nodes:

                    continue


                if node_name in agent_order:

                    completed_nodes.add(
                        node_name
                    )


                # ----------------------------------------------
                # COORDINATOR
                # ----------------------------------------------

                if node_name == "coordinator":

                    coordinator_status.success(
                        "🧠 Coordinator Agent: Completed"
                    )

                    web_status.warning(
                        "🔎 Web Researcher: Working..."
                    )

                    status_text.info(
                        "🔎 Web Researcher is searching for information..."
                    )


                # ----------------------------------------------
                # WEB RESEARCHER
                # ----------------------------------------------

                elif node_name == "web_researcher":

                    web_status.success(
                        "🔎 Web Researcher: Completed"
                    )

                    academic_status.warning(
                        "📚 Academic Researcher: Working..."
                    )

                    status_text.info(
                        "📚 Academic Researcher is searching for papers..."
                    )


                # ----------------------------------------------
                # ACADEMIC RESEARCHER
                # ----------------------------------------------

                elif node_name == "academic_researcher":

                    academic_status.success(
                        "📚 Academic Researcher: Completed"
                    )

                    fact_status.warning(
                        "🔍 Fact Checker: Working..."
                    )

                    status_text.info(
                        "🔍 Fact Checker is verifying research findings..."
                    )


                # ----------------------------------------------
                # FACT CHECKER
                # ----------------------------------------------

                elif node_name == "fact_checker":

                    fact_status.success(
                        "🔍 Fact Checker: Completed"
                    )

                    summary_status.warning(
                        "📝 Summarizer: Working..."
                    )

                    status_text.info(
                        "📝 Summarizer is preparing the research summary..."
                    )


                # ----------------------------------------------
                # SUMMARIZER
                # ----------------------------------------------

                elif node_name == "summarizer":

                    summary_status.success(
                        "📝 Summarizer: Completed"
                    )

                    report_status.warning(
                        "📄 Report Generator: Working..."
                    )

                    status_text.info(
                        "📄 Report Generator is creating the PDF..."
                    )


                # ----------------------------------------------
                # REPORT GENERATOR
                # ----------------------------------------------

                elif node_name == "report_generator":

                    report_status.success(
                        "📄 Report Generator: Completed"
                    )

                    status_text.success(
                        "✅ All research agents completed!"
                    )


                # ----------------------------------------------
                # UPDATE PROGRESS
                # ----------------------------------------------

                completed_count = len(
                    completed_nodes
                )


                percentage = int(
                    (
                        completed_count /
                        total_agents
                    ) * 100
                )


                progress.progress(
                    percentage
                )


        # --------------------------------------------------
        # FINAL PROGRESS
        # --------------------------------------------------

        progress.progress(100)

        status_text.success(
            "🎉 Research completed successfully!"
        )


        # ==================================================
        # RESEARCH STATISTICS
        # ==================================================

        st.divider()

        st.subheader(
            "📊 Research Statistics"
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Research Tasks",
                len(
                    result.get(
                        "subtasks",
                        []
                    )
                )
            )


        with col2:

            st.metric(
                "Web Sources",
                len(
                    result.get(
                        "web_results",
                        []
                    )
                )
            )


        with col3:

            st.metric(
                "Academic Papers",
                len(
                    result.get(
                        "academic_results",
                        []
                    )
                )
            )


        # ==================================================
        # RESEARCH SUMMARY
        # ==================================================

        st.divider()

        st.subheader(
            "📝 Research Summary"
        )


        summary = result.get(
            "summary",
            ""
        )


        if summary:

            st.markdown(
                summary
            )

        else:

            st.info(
                "No research summary was generated."
            )


        # ==================================================
        # RESEARCH TASKS
        # ==================================================

        with st.expander(
            "🧠 Research Tasks"
        ):

            subtasks = result.get(
                "subtasks",
                []
            )


            if subtasks:

                for index, task in enumerate(
                    subtasks,
                    start=1
                ):

                    st.write(
                        f"{index}. {task}"
                    )

            else:

                st.info(
                    "No research tasks were generated."
                )


        # ==================================================
        # FACT CHECKING
        # ==================================================

        with st.expander(
            "🔍 Fact Check Results"
        ):

            verified_claims = result.get(
                "verified_claims",
                []
            )


            if not verified_claims:

                st.info(
                    "No fact-check results were found."
                )

            elif isinstance(
                verified_claims,
                list
            ):

                for index, claim in enumerate(
                    verified_claims,
                    start=1
                ):

                    st.write(
                        f"{index}. {claim}"
                    )

            else:

                st.markdown(
                    str(verified_claims)
                )


        # ==================================================
        # WEB SOURCES
        # ==================================================

        with st.expander(
            "🔎 Web Sources"
        ):

            web_results = result.get(
                "web_results",
                []
            )


            if not web_results:

                st.info(
                    "No web sources were found."
                )

            else:

                for index, item in enumerate(
                    web_results,
                    start=1
                ):

                    st.markdown(
                        f"### {index}. "
                        f"{item.get('title', 'Untitled Source')}"
                    )


                    st.write(
                        item.get(
                            "snippet",
                            "No description available."
                        )
                    )


                    url = item.get(
                        "url",
                        ""
                    )


                    if url:

                        st.link_button(
                            "🔗 Open Source",
                            url
                        )


                    st.divider()


        # ==================================================
        # ACADEMIC SOURCES
        # ==================================================

        with st.expander(
            "📚 Academic Papers"
        ):

            academic_results = result.get(
                "academic_results",
                []
            )


            if not academic_results:

                st.info(
                    "No academic papers were found."
                )

            else:

                for index, paper in enumerate(
                    academic_results,
                    start=1
                ):

                    st.markdown(
                        f"### {index}. "
                        f"{paper.get('title', 'Untitled Paper')}"
                    )


                    authors = paper.get(
                        "authors",
                        []
                    )


                    if isinstance(
                        authors,
                        list
                    ):

                        authors_text = ", ".join(
                            authors
                        )

                    else:

                        authors_text = str(
                            authors
                        )


                    st.write(
                        f"**Authors:** "
                        f"{authors_text}"
                    )


                    st.write(
                        f"**Published:** "
                        f"{paper.get('published', 'Unknown')}"
                    )


                    st.write(
                        paper.get(
                            "summary",
                            "No summary available."
                        )
                    )


                    url = paper.get(
                        "url",
                        ""
                    )


                    if url:

                        st.link_button(
                            "🔗 View Paper",
                            url
                        )


                    st.divider()


        # ==================================================
        # DOWNLOAD REPORT
        # ==================================================

        st.divider()

        st.subheader(
            "📄 Research Report"
        )


        report_file = result.get(
            "final_report",
            ""
        )


        if report_file:

            report_path = Path(
                report_file
            )


            if report_path.exists():

                with open(
                    report_path,
                    "rb"
                ) as file:

                    st.download_button(
                        label=(
                            "📥 Download Research Report"
                        ),
                        data=file,
                        file_name=(
                            "research_report.pdf"
                        ),
                        mime="application/pdf"
                    )

            else:

                st.error(
                    "Report file could not be found."
                )

        else:

            st.info(
                "No report was generated."
            )


    # ==================================================
    # ERROR HANDLING
    # ==================================================

    except Exception as e:

        status_text.error(
            f"❌ Research failed: {e}"
        )

        st.exception(e)

        st.stop()