from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .agents.article_matcher import ArticleMatcherAgent
from .agents.conflict_detector import ConflictDetectorAgent
from .agents.domain_classifier import DomainClassifierAgent
from .agents.final_agent import FinalComplianceAgent
from .agents.ingestion import IngestionAgent
from .agents.llm_reasoner import LLMReasonerAgent
from .config import ProjectConfig, load_project_config
from .data_models import ComplianceState


@dataclass
class CompliancePipeline:
    config: ProjectConfig
    agents: List = field(default_factory=list)

    @classmethod
    def build_default(
        cls,
        config: ProjectConfig,
        use_llm: bool = False,
        llm_model: str = "llama-3.3-70b-versatile",
    ) -> "CompliancePipeline":
        agents: List = [
            IngestionAgent(),
            DomainClassifierAgent(config.rulebook),
            ArticleMatcherAgent(config),
            ConflictDetectorAgent(config, use_llm=use_llm),
        ]
        if use_llm:
            agents.append(LLMReasonerAgent(enabled=True, model=llm_model))
        agents.append(FinalComplianceAgent())
        return cls(config=config, agents=agents)

    def evaluate(self, policy_path: str | Path, policy_id: str | None = None) -> ComplianceState:
        source = Path(policy_path)
        if not source.exists():
            raise FileNotFoundError(f"Policy path not found: {policy_path}")
        state = ComplianceState(
            policy_id=policy_id or source.stem,
            source=str(source),
            raw_text="",
        )
        for agent in self.agents:
            state = agent(state)
        return state
    
    def evaluate_text(self, policy_text: str, policy_id: str = "user_input") -> ComplianceState:
        """Evaluate policy text directly without requiring a file."""
        from tempfile import NamedTemporaryFile
        import os
        
        # Create a temporary file with the policy text
        with NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp_file:
            tmp_file.write(policy_text)
            tmp_path = tmp_file.name
        
        try:
            state = self.evaluate(tmp_path, policy_id=policy_id)
            # Update source to indicate it was from text input
            state.source = "text_input"
            return state
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


@dataclass
class LangGraphPipeline:
    config: ProjectConfig
    use_llm: bool = False
    llm_model: str = "llama-3.3-70b-versatile"
    _app: Optional[object] = field(init=False, default=None)
    _llm_agent: Optional[LLMReasonerAgent] = field(init=False, default=None)

    def __post_init__(self) -> None:
        self._llm_agent = LLMReasonerAgent(enabled=self.use_llm, model=self.llm_model) if self.use_llm else None
        self._app = self._build_graph()

    def _build_graph(self):
        try:
            from langgraph.graph import END, StateGraph
        except ImportError as exc:  # noqa: F401
            raise ImportError("langgraph is not installed. Please add it to requirements.") from exc

        def wrap(agent):
            def node(state_dict):
                state: ComplianceState = state_dict["state"]
                updated = agent(state)
                return {"state": updated}

            return node

        graph = StateGraph(dict)
        graph.add_node("ingestion", wrap(IngestionAgent()))
        graph.add_node("classification", wrap(DomainClassifierAgent(self.config.rulebook)))
        graph.add_node("retrieval", wrap(ArticleMatcherAgent(self.config)))
        graph.add_node("conflict", wrap(ConflictDetectorAgent(self.config, use_llm=self.use_llm)))

        if self._llm_agent:
            graph.add_node("llm_reason", wrap(self._llm_agent))

        graph.add_node("final", wrap(FinalComplianceAgent()))

        graph.set_entry_point("ingestion")
        graph.add_edge("ingestion", "classification")
        graph.add_edge("classification", "retrieval")
        graph.add_edge("retrieval", "conflict")

        if self._llm_agent:
            def route_llm(state_dict):
                state: ComplianceState = state_dict["state"]
                if state.conflicts and self._llm_agent and self._llm_agent.enabled:
                    return "llm_reason"
                return "final"

            graph.add_conditional_edges(
                "conflict",
                route_llm,
                {
                    "llm_reason": "llm_reason",
                    "final": "final",
                },
            )
            graph.add_edge("llm_reason", "final")
        else:
            graph.add_edge("conflict", "final")

        graph.set_finish_point("final")
        return graph.compile()

    def evaluate(self, policy_path: str | Path, policy_id: str | None = None) -> ComplianceState:
        source = Path(policy_path)
        if not source.exists():
            raise FileNotFoundError(f"Policy path not found: {policy_path}")
        state = ComplianceState(
            policy_id=policy_id or source.stem,
            source=str(source),
            raw_text="",
        )
        result = self._app.invoke({"state": state})
        return result["state"]
    
    def evaluate_text(self, policy_text: str, policy_id: str = "user_input") -> ComplianceState:
        """Evaluate policy text directly without requiring a file."""
        from tempfile import NamedTemporaryFile
        import os
        
        # Create a temporary file with the policy text
        with NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp_file:
            tmp_file.write(policy_text)
            tmp_path = tmp_file.name
        
        try:
            state = self.evaluate(tmp_path, policy_id=policy_id)
            # Update source to indicate it was from text input
            state.source = "text_input"
            return state
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


def load_pipeline(
    root_dir: str | Path = ".",
    engine: str = "sequential",
    use_llm: bool = False,
    llm_model: str = "llama-3.3-70b-versatile",
):
    config = load_project_config(root_dir=root_dir)
    if engine == "langgraph":
        return LangGraphPipeline(config=config, use_llm=use_llm, llm_model=llm_model)
    return CompliancePipeline.build_default(config, use_llm=use_llm, llm_model=llm_model)

