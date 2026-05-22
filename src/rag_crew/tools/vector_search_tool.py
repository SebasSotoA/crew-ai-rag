from crewai.tools import BaseTool
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from pydantic import Field

from rag_crew.vector_store import CHROMA_DB_PATH


class VectorSearchTool(BaseTool):
    name: str = "vector_search"
    description: str = "Searches the vector store for chunks relevant to the query."
    k: int = Field(default=5)

    def _run(self, query: str) -> str:
        db = Chroma(
            persist_directory=str(CHROMA_DB_PATH),
            embedding_function=OpenAIEmbeddings(),
        )
        results = db.similarity_search(query, k=self.k)
        return "\n\n---\n\n".join([doc.page_content for doc in results])