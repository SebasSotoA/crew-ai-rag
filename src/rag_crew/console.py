#!/usr/bin/env python
"""Interactive console for exploring, rebuilding, and querying the RAG index."""

from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# Avoid Windows console encoding errors on PDF special characters
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

from rag_crew.main import run_query
from rag_crew.vector_store import (
    CHROMA_DB_PATH,
    DOCUMENTS_FOLDER,
    build_vector_store,
    chunk_count,
    inspect_vector_store,
)


def _confirm(prompt: str) -> bool:
    answer = input(f"{prompt} [y/N]: ").strip().lower()
    return answer in {"y", "yes"}


def _print_header() -> None:
    count = chunk_count()
    index_status = f"{count} chunks" if count else "not built"
    print("\n=== RAG Console ===")
    print(f"Documents folder: {DOCUMENTS_FOLDER}")
    print(f"Vector store:     {CHROMA_DB_PATH} ({index_status})")
    print()


def _menu_explore() -> None:
    if chunk_count() == 0:
        print("\nNo vector index found. Add PDFs to your documents folder and run Rebuild.\n")
        return

    inspect_vector_store()
    query = input("\nSearch preview (Enter to skip): ").strip()
    if query:
        inspect_vector_store(query=query)
    print()


def _menu_rebuild() -> None:
    print(f"\nPDFs will be read from: {DOCUMENTS_FOLDER}")
    print("Place your PDF files in that folder before continuing.")
    if not _confirm(
        f"\nThis will delete the current index and rebuild from {DOCUMENTS_FOLDER}"
    ):
        print("Rebuild cancelled.\n")
        return

    try:
        result = build_vector_store()
        if result is None:
            print("Rebuild failed — no PDFs found or an error occurred.\n")
        else:
            print("Rebuild complete.\n")
    except Exception as exc:
        print(f"Rebuild failed: {exc}\n")


def _menu_ask() -> None:
    if chunk_count() == 0:
        print(
            "\nNo vector index found. Run Rebuild first after adding PDFs to "
            f"{DOCUMENTS_FOLDER}.\n"
        )
        return

    print("\nAsk questions about your documents.")
    print("Each question runs the full RAG crew (may take a minute).")
    print("Type 'back' to return to the main menu.\n")

    while True:
        question = input("Question> ").strip()
        if not question:
            continue
        if question.lower() in {"back", "exit", "quit"}:
            print()
            break

        try:
            result = run_query(question)
            print("\n=== Answer ===")
            print(result)
            print()
        except Exception as exc:
            print(f"\nError: {exc}\n")


def main() -> None:
    actions = {
        "1": ("Explore vector database", _menu_explore),
        "2": ("Rebuild index (clear + re-index PDFs)", _menu_rebuild),
        "3": ("Ask a question (RAG crew)", _menu_ask),
        "4": ("Exit", None),
    }

    while True:
        _print_header()
        for key, (label, _) in actions.items():
            print(f"  {key}) {label}")

        choice = input("\nChoice: ").strip()
        if choice == "4":
            print("Goodbye.")
            break

        action = actions.get(choice)
        if action is None:
            print("Invalid choice. Enter 1, 2, 3, or 4.\n")
            continue

        _, handler = action
        if handler:
            handler()


if __name__ == "__main__":
    main()
