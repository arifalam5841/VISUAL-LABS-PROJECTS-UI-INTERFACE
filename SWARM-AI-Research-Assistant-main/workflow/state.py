from typing import TypedDict


class ResearchState(TypedDict):

    query: str

    subtasks: list

    web_results: list

    academic_results: list

    findings: list

    verified_claims: list

    summary: str

    final_report: str