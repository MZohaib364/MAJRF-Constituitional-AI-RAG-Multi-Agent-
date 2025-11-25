# Constitutional Compliance Checker (Agentic AI Project)

An agentic AI system that evaluates Pakistani policy drafts against constitutional provisions, legislative lists, and jurisdictional rules. The system satisfies NCEAC’s Complex Computing Problem criteria via multi-agent orchestration (LangGraph), Groq-backed LLM reasoning, automated tool use (document ingestion, vector retrieval), and an experimentation harness for research-grade evaluation.

## Features
- **Multi-agent architecture**: Ingestion, Domain Classification, Article Retrieval, Rule-based Conflict Detection, Groq LLM Reasoner, and Final Compliance agents coordinated via a LangGraph state machine.
- **LLM reasoning**: Optional Groq `llama-3.3-70b-versatile` (configurable) produces structured legal analysis, confidence, and remediation guidance.
- **Knowledge-grounded retrieval**: ChromaDB vector store with sentence-transformers embeddings for semantic similarity search over constitutional articles, legislative lists, and rulebook metadata (RAG implementation).
- **Jurisdictional rule engine**: Enforces federal vs provincial authority (Articles 141–149) and fundamental rights (Articles 8–28) with severity/confidence estimates.
- **Experimentation toolkit**: `scripts/run_experiments.py` benchmarks predictions against labeled cases and logs precision/recall/F1/coverage for paper-ready analysis.
- **Ethics & governance hooks**: Logs every agent decision, supports human-in-the-loop review, and surfaces transparency artifacts (citations, confidence, recommended mitigations).
- **Beautiful Web Interface**: Streamlit-based frontend with file upload, text input, and comprehensive result visualization.

## Quick Start

### Web Interface (Recommended)
```powershell
# Activate existing venv
& C:/Users/hp/OneDrive/Desktop/Agentic-Project/venv/Scripts/Activate.ps1

# Install/refresh dependencies
pip install -r requirements.txt

# Launch Streamlit web interface
streamlit run app.py
# OR
python run_app.py
```

The web interface provides:
- 📄 Upload policy files or type text directly
- 📊 Beautiful visualization of results
- ⚖️ Detailed conflict analysis
- 📚 Relevant articles with snippets
- 💡 Recommended actions
- 💾 Export results as JSON

### Command Line Interface
```powershell
# Activate existing venv (as requested)
& C:/Users/hp/OneDrive/Desktop/Agentic-Project/venv/Scripts/Activate.ps1

# Install/refresh dependencies
pip install -r requirements.txt

# Run sequential pipeline on sample policies
python main.py --input Data/samples --output outputs --pipeline sequential

# Run LangGraph multi-agent pipeline with Groq LLM reasoning
# API key is loaded from groq.env file (already configured)
python main.py --input Data/samples --pipeline langgraph --use-llm --groq-model llama-3.3-70b-versatile
```

Outputs are stored under `outputs/<case>.json`, containing segments, retrieved articles, rule-engine conflicts, LLM reasoning, and the final compliance verdict.

## Agent Graph
```
Ingestion → DomainClassifier → ArticleMatcher → ConflictDetector ─┬─→ FinalCompliance
                                                                  └─→ LLMReasoner → FinalCompliance
```
- Implemented with `langgraph` to explicitly model branching under uncertainty (LLM path triggers only when conflicts exist and Groq is available).
- Each agent logs its status/status codes to `ComplianceState.log` for audit trails.

## Experimentation & Metrics
```
python scripts/run_experiments.py ^
  --input Data/samples ^
  --labels Data/samples/labels.json ^
  --pipeline langgraph ^
  --use-llm ^
  --output experiments/results
```
Generates precision/recall/F1/coverage scores via `src/evaluation/metrics.py` and writes a timestamped JSON report (ready for inclusion in the IEEE paper’s evaluation section). Extend `Data/samples` with more cases + labels to scale experiments.

## Extensibility Hooks
- **LLM Swaps**: change `--groq-model` or modify `src/agents/llm_reasoner.py` to plug in tool-calling strategies, chain-of-thought, or debate-style multi-agent prompts.
- **Alternative orchestrators**: `src/orchestrator.py` exposes both sequential and LangGraph pipelines; you can add AutoGen or CrewAI variants without touching agent logic.
- **Ethics & auditing**: integrate bias/fairness checkers as additional nodes before `FinalComplianceAgent`, or connect to human review queues for high-risk outcomes.
- **Research instrumentation**: `ComplianceState.to_dict()` makes it easy to stream intermediate states to a data lake for further analysis (e.g., prompt ablations, confidence calibration).

## Test Cases

The `Data/samples/` directory contains 7 test cases:
- **case1.txt**: Non-compliant - Biometric data collection without judicial authorization (Article 14 violation)
- **case2.txt**: Non-compliant - Movement restrictions based on ethnicity (Article 15, 25 violations)
- **case3.txt**: Non-compliant - Ban on online criticism without legal basis (Article 19, 19A violations)
- **case4.txt**: Compliant - Digital Dignity Act with proper safeguards (Articles 14, 10A compliance)
- **case5.txt**: Compliant - Healthcare policy within provincial jurisdiction (no conflicts)
- **case6.txt**: Compliant - ICT building code with proper appeal mechanisms (federal jurisdiction)
- **case7.txt**: Non-compliant - Emergency detention order without due process (Article 10A violation)

Ground truth labels are in `Data/samples/labels.json` for evaluation metrics.

## Repository Layout (key paths)
- `app.py` – Streamlit web interface for interactive policy analysis
- `run_app.py` – Simple script to launch the Streamlit app
- `Data/constitution/` – curated article snippets (use instead of full PDF).
- `Data/rules/rulebook.json` – domain mappings, legislative lists, conflict rules.
- `Data/samples/` – test cases (compliant and non-compliant) with labels.
- `groq.env` – Groq API key (loaded automatically via python-dotenv).
- `src/agents/` – modular agent implementations (including `LLMReasonerAgent`).
- `src/llm/groq_client.py` – Groq API wrapper.
- `src/knowledge_base.py` – ChromaDB RAG implementation with sentence-transformers.
- `src/orchestrator.py` – sequential + LangGraph pipelines with text evaluation support.
- `src/evaluation/metrics.py` – precision/recall/F1/coverage computations.
- `scripts/run_experiments.py` – reproducible benchmarking.


