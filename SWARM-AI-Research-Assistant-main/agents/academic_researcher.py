import arxiv


def academic_researcher_agent(state):

    subtasks = state["subtasks"]

    all_results = []

    for task in subtasks:

        print(f"\n📚 Searching academic papers for: {task}")

        try:

            search = arxiv.Search(
                query=task,
                max_results=3,
                sort_by=arxiv.SortCriterion.Relevance
            )

            client = arxiv.Client()

            results = client.results(search)

            for paper in results:

                all_results.append({
                    "task": task,
                    "title": paper.title,
                    "authors": [
                        author.name
                        for author in paper.authors
                    ],
                    "summary": paper.summary,
                    "published": str(paper.published),
                    "url": paper.entry_id
                })

        except Exception as e:

            print(
                f"Academic search error: {e}"
            )

    return {
        "academic_results": all_results
    }