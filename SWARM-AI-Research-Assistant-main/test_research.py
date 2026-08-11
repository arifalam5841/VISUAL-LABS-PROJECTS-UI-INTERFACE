from workflow.graph import create_research_graph


graph = create_research_graph()


query = input(
    "Enter your research topic: "
)


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


print("\n🚀 Starting AI Research Swarm...")


result = graph.invoke(initial_state)


# --------------------------------------------------
# RESEARCH SUBTASKS
# --------------------------------------------------

print("\n" + "=" * 60)
print("RESEARCH SUBTASKS")
print("=" * 60)


for task in result["subtasks"]:

    print(task)


# --------------------------------------------------
# FACT CHECK RESULTS
# --------------------------------------------------

print("\n" + "=" * 60)
print("FACT CHECK RESULTS")
print("=" * 60)

print(
    result["verified_claims"]
)


# --------------------------------------------------
# FINAL SUMMARY
# --------------------------------------------------

print("\n" + "=" * 60)
print("RESEARCH SUMMARY")
print("=" * 60)

print(
    result["summary"]
)

# --------------------------------------------------
# FINAL REPORT
# --------------------------------------------------

print("\n" + "=" * 60)
print("FINAL REPORT")
print("=" * 60)

print(
    f"Report saved at: "
    f"{result['final_report']}"
)