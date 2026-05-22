#!/usr/bin/env python
import warnings
from rag_crew.crew import RagCrew

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def run_query(query: str):
    """Run the RAG crew for a single question and return the result."""
    return RagCrew().crew().kickoff(inputs={"query": query})


def run():
    """Run the crew with the default query from main."""
    query = "How many vacation days do I get after 2 years?"
    try:
        result = run_query(query)
        print("\n\n=== FINAL ANSWER ===")
        print(result)
    except Exception as e:
        raise RuntimeError(f"An error occurred while running the crew: {e}") from e


if __name__ == "__main__":
    run()
