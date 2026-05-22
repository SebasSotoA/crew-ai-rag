from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from rag_crew.tools.vector_search_tool import VectorSearchTool


@CrewBase
class RagCrew:
    """RagCrew crew."""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def retriever_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["retriever_agent"],  # type: ignore
            tools=[VectorSearchTool()],
            verbose=True,
        )

    @agent
    def analyst_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["analyst_agent"],  # type: ignore
            verbose=True,
        )

    @agent
    def supervisor_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["supervisor_agent"],  # type: ignore
            verbose=True,
        )

    @task
    def retrieve_task(self) -> Task:
        return Task(config=self.tasks_config["retrieve_task"])

    @task
    def analyze_task(self) -> Task:
        return Task(config=self.tasks_config["analyze_task"])

    @task
    def respond_task(self) -> Task:
        return Task(config=self.tasks_config["respond_task"])

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )