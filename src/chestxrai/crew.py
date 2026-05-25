from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from tools.triage_tool import XrayTool
from tools.guideline_tool import GuidelineTool
from tools.vlm_tool import VLMReviewTool

llm = LLM(
    model="ollama/mistral:7b",
    base_url="http://localhost:11434",
)

# Stronger model for the orchestrator — needed to reliably delegate tasks
# to sub-agents rather than answering them directly.
manager_llm = LLM(
    model="ollama/qwen2.5:14b",
    base_url="http://localhost:11434",
)

@CrewBase
class ChestXRAICrew():

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def radiologist(self) -> Agent:
        return Agent(
            config=self.agents_config['radiologist'],
            verbose=True,
            llm=llm,
            tools=[XrayTool()],
        )

    @agent
    def vlm_reviewer(self) -> Agent:
        return Agent(
            config=self.agents_config['vlm_reviewer'],
            verbose=True,
            llm=llm,
            tools=[VLMReviewTool()],
        )

    @agent
    def clinical_advisor(self) -> Agent:
        return Agent(
            config=self.agents_config['clinical_advisor'],
            verbose=True,
            llm=llm,
            tools=[GuidelineTool()],
        )

    @agent
    def report_generator(self) -> Agent:
        return Agent(
            config=self.agents_config['report_generator'],
            verbose=True,
            llm=llm,
        )

    @task
    def analyze_xray(self) -> Task:
        return Task(config=self.tasks_config['analyze_xray'])

    @task
    def vlm_review(self) -> Task:
        return Task(config=self.tasks_config['vlm_review'])

    @task
    def retrieve_guidelines(self) -> Task:
        return Task(config=self.tasks_config['retrieve_guidelines'])

    @task
    def generate_report(self) -> Task:
        return Task(config=self.tasks_config['generate_report'])

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.hierarchical,
            manager_llm=manager_llm,
            verbose=True,
        )
