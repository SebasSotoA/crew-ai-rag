#!/usr/bin/env python
import warnings

from rag_crew.crew import RagCrew

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def run():
    inputs = {"query": "What are AI applications in cloud security according to the document?"}
    try:
        result = RagCrew().crew().kickoff(inputs=inputs)
        print("\n\n=== FINAL ANSWER ===")
        print(result)
    except Exception as e:
        raise RuntimeError(f"An error occurred while running the crew: {e}") from e

if __name__ == "__main__":
    run()