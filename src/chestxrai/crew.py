import queue as _q
import re as _re

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from tools.triage_tool import XrayTool
from tools.guideline_tool import GuidelineTool
from tools.vlm_tool import VLMReviewTool
from llm_config import agent_llm, manager_llm

# ── Shared log queue ──────────────────────────────────────────
# Set from api/main.py before kickoff; callbacks read it at call-time.
_active_log_q: _q.SimpleQueue | None = None

_ANSI = _re.compile(r'\x1b\[[0-9;]*[mGKHF]')

def _clean(s: str) -> str:
    return _ANSI.sub('', s).strip()

def _extract(step) -> str | None:
    if isinstance(step, str):
        s = _clean(step)
        return s if len(s) > 4 else None
    for attr in ('thought', 'log'):
        val = getattr(step, attr, None)
        if isinstance(val, str):
            s = _clean(val)
            if len(s) > 4:
                return s
    if hasattr(step, 'tool'):
        tool = step.tool or ''
        inp  = _clean(str(getattr(step, 'tool_input', '') or ''))
        inp_part = f" ← {inp}" if inp and inp not in ('{}', 'null', 'None', '') else ''
        return f"Using: {tool}{inp_part}"
    if hasattr(step, 'return_values'):
        rv  = step.return_values
        out = rv.get('output', '') if isinstance(rv, dict) else str(rv)
        s   = _clean(str(out))
        return s if len(s) > 4 else None
    s = _clean(str(step))
    return s if len(s) > 4 else None

def _make_cb(tag: str):
    def cb(step_output):
        q = _active_log_q
        if q is None:
            return
        text = _extract(step_output)
        if text:
            q.put(f"{tag} {text}")
    return cb


@CrewBase
class ChestXRAICrew():

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def radiologist(self) -> Agent:
        return Agent(
            config=self.agents_config['radiologist'],
            verbose=True,
            llm=agent_llm,
            tools=[XrayTool()],
            step_callback=_make_cb('[Radiologist]'),
        )

    @agent
    def vlm_reviewer(self) -> Agent:
        return Agent(
            config=self.agents_config['vlm_reviewer'],
            verbose=True,
            llm=agent_llm,
            tools=[VLMReviewTool()],
            step_callback=_make_cb('[VLM Review]'),
        )

    @agent
    def clinical_advisor(self) -> Agent:
        return Agent(
            config=self.agents_config['clinical_advisor'],
            verbose=True,
            llm=agent_llm,
            tools=[GuidelineTool()],
            step_callback=_make_cb('[Clinical Advisor]'),
        )

    @agent
    def report_generator(self) -> Agent:
        return Agent(
            config=self.agents_config['report_generator'],
            verbose=True,
            llm=agent_llm,
            step_callback=_make_cb('[Report Generator]'),
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
            step_callback=_make_cb('[Orchestrator]'),
        )
