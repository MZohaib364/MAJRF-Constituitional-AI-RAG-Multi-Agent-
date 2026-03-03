# MAJRF: Multi-Agent Jurisdictional Reasoning Framework 🤖⚖️

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-green.svg)](https://langchain-ai.github.io/langgraph/)
[![RAG](https://img.shields.io/badge/RAG-ChromaDB-orange.svg)](https://www.trychroma.com/)
[![LLM](https://img.shields.io/badge/LLM-Groq%20Llama--3.3--70B-red.svg)](https://groq.com/)

> **Research Implementation**: A novel Multi-Agent Jurisdictional Reasoning Framework for Constitutional Compliance Analysis of Policy Scenarios

An advanced **agentic AI system** that leverages **multi-agent orchestration**, **retrieval-augmented generation (RAG)**, and **hybrid reasoning** to automatically analyze policy documents for constitutional compliance. Built with cutting-edge AI/ML technologies including LangGraph state machines, sentence-transformers embeddings, ChromaDB vector stores, and Groq LLM reasoning.

## 🎯 Research Contributions

This implementation demonstrates several advanced AI/ML concepts:

- **Multi-Agent Architecture**: Six specialized agents orchestrated via LangGraph state machines
- **Hybrid Reasoning**: Combines embedding-based semantic matching with LLM logical reasoning  
- **RAG Implementation**: ChromaDB vector store with sentence-transformers for constitutional article retrieval
- **Anti-Hallucination Framework**: Grounded prompts, explicit constraints, and post-processing validation
- **Conditional Routing**: Cost-optimized LLM usage with 50% reduction in API calls
- **Semantic Classification**: Domain classification using cosine similarity on sentence embeddings

**Performance Metrics**: Precision: 0.90 | Recall: 0.80 | F1-Score: 0.85

## 🏗️ System Architecture

### Multi-Agent Pipeline
```
📄 Ingestion → 🏷️ Domain Classifier → 🔍 Article Matcher → ⚠️ Conflict Detector ─┬─→ 📋 Final Compliance
                                                                                  └─→ 🧠 LLM Reasoner → 📋 Final Compliance
```

### Agent Specifications

| Agent | Technology Stack | Responsibility |
|-------|-----------------|----------------|
| **Ingestion Agent** | Document Processing | Text extraction, normalization, segmentation |
| **Domain Classifier** | sentence-transformers, cosine similarity | Semantic domain classification (cybersecurity, healthcare, etc.) |
| **Article Matcher** | ChromaDB, RAG, all-MiniLM-L6-v2 | Constitutional article retrieval via semantic search |
| **Conflict Detector** | Embedding-based pattern matching | Violation detection using semantic similarity (threshold: 0.42) |
| **LLM Reasoner** | Groq Llama-3.3-70B, conditional routing | Deep contextual analysis with anti-hallucination measures |
| **Final Compliance** | Result aggregation | Comprehensive diagnosis and remediation suggestions |

## 🚀 Key Features

### Advanced AI/ML Techniques
- **🔗 LangGraph State Machines**: Sophisticated agent orchestration with conditional routing
- **🎯 RAG with ChromaDB**: Vector-based constitutional article retrieval 
- **🧮 Sentence Transformers**: all-MiniLM-L6-v2 for semantic embeddings
- **🛡️ Anti-Hallucination**: Grounded prompts, explicit constraints, post-processing validation
- **📊 Hybrid Detection**: Embedding similarity + LLM reasoning for robust conflict detection
- **💰 Cost Optimization**: Conditional LLM usage reduces API calls by ~50%

### Technical Capabilities
- **Multi-format Document Processing**: PDF, TXT with intelligent text extraction
- **Semantic Domain Classification**: Automated policy categorization using embedding similarity
- **Constitutional Knowledge Base**: Structured rulebook with Articles 141-149, Fourth Schedule
- **Confidence Scoring**: Probabilistic violation detection with severity classification
- **Audit Trail**: Complete execution logging for transparency and debugging
- **Human-in-the-Loop**: Review capabilities with interpretable explanations

## 🛠️ Installation & Setup

### Prerequisites
```bash
Python 3.8+
pip install -r requirements.txt
```

### Environment Configuration
```bash
# Set up Groq API key
echo "GROQ_API_KEY=your_groq_api_key_here" > groq.env
```

## 🎮 Usage

### Web Interface (Streamlit)
```bash
# Launch interactive web application
streamlit run app.py
# OR
python run_app.py
```

**Features:**
- 📤 File upload (PDF/TXT) or direct text input
- 📊 Real-time compliance analysis with visualizations
- ⚖️ Detailed constitutional conflict breakdown
- 📚 Retrieved articles with relevance scores
- 💡 AI-generated remediation suggestions
- 💾 Export results as JSON

### Command Line Interface
```bash
# Sequential pipeline (embedding-only)
python main.py --input Data/samples --output outputs --pipeline sequential

# Multi-agent pipeline with LLM reasoning
python main.py --input Data/samples --pipeline langgraph --use-llm --groq-model llama-3.3-70b-versatile

# Batch processing with evaluation metrics
python scripts/run_experiments.py --input Data/samples --labels Data/samples/labels.json --pipeline langgraph --use-llm
```

## 📊 Evaluation & Metrics

### Research-Grade Benchmarking
```bash
python scripts/run_experiments.py \
  --input Data/samples \
  --labels Data/samples/labels.json \
  --pipeline langgraph \
  --use-llm \
  --output experiments/results
```

**Evaluation Metrics:**
- Precision: 0.90
- Recall: 0.80  
- F1-Score: 0.85
- Domain Classification Accuracy: 95%+
- Hallucination Rate: <5% (with anti-hallucination measures)

### Test Cases
The system includes 7 comprehensive test cases covering:
- ❌ **Non-compliant**: Biometric data collection, movement restrictions, speech limitations
- ✅ **Compliant**: Healthcare policies, building codes, digital rights frameworks
- 🎯 **Edge Cases**: Emergency powers, jurisdictional boundaries, fundamental rights

## 🔬 Technical Deep Dive

### Embedding-Based Semantic Matching
```python
# Domain classification using sentence-transformers
embeddings = sentence_transformer.encode(policy_text)
similarity = cosine_similarity(embeddings, domain_embeddings)
domain = domains[np.argmax(similarity)]
```

### RAG Implementation
```python
# Constitutional article retrieval
query_embedding = embedder.encode(policy_segment)
results = chroma_collection.query(
    query_embeddings=[query_embedding],
    n_results=5
)
```

### Anti-Hallucination Framework
- **Grounded Prompts**: LLM receives actual constitutional text, not just references
- **Explicit Constraints**: Prohibited from citing non-retrieved articles
- **Post-Processing**: Invalid citations automatically filtered
- **Confidence Thresholds**: Only high-confidence violations (>0.7) reported

## 🔧 Extensibility

### Adding New Agents
```python
class CustomAgent(BaseAgent):
    def process(self, state: ComplianceState) -> ComplianceState:
        # Custom logic here
        return state
```

### LLM Integration
- **Groq**: llama-3.3-70b-versatile (default)
- **OpenAI**: GPT-4, GPT-3.5-turbo
- **Anthropic**: Claude models
- **Local**: Ollama, vLLM deployments

### Vector Store Options
- **ChromaDB** (default)
- **Pinecone**
- **Weaviate**
- **FAISS**

## 📁 Project Structure

```
MAJRF/
├── 📱 app.py                    # Streamlit web interface
├── 🚀 run_app.py               # App launcher
├── 🎯 main.py                  # CLI entry point
├── 📊 Data/
│   ├── constitution/           # Constitutional articles
│   ├── rules/rulebook.json    # Domain mappings & conflict rules
│   └── samples/               # Test cases with ground truth
├── 🤖 src/
│   ├── agents/                # Multi-agent implementations
│   ├── llm/groq_client.py    # Groq API integration
│   ├── knowledge_base.py     # ChromaDB RAG system
│   ├── orchestrator.py       # LangGraph pipeline
│   └── evaluation/metrics.py # Performance evaluation
├── 🧪 scripts/
│   └── run_experiments.py    # Benchmarking suite
└── 📋 requirements.txt        # Dependencies
```

## 🎓 Academic Context

This implementation is based on the research paper:
**"A Multi-Agent Jurisdictional Reasoning Framework for Constitutional Compliance Analysis of Policy Scenarios"**

**Authors**: Abdul Moiz Rana, Muhammad Zohaib, Danish Ali  
**Institution**: FAST-NUCES, Islamabad, Pakistan  
**Domain**: Legal AI, Multi-Agent Systems, Retrieval-Augmented Generation

## 🤝 Contributing

Contributions welcome! Areas of interest:
- Additional constitutional frameworks (India, Malaysia, Nigeria)
- Advanced reasoning techniques (Chain-of-Thought, Tree-of-Thoughts)
- Multi-modal document processing (images, tables)
- Bias detection and fairness metrics
- Real-time policy monitoring systems

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangGraph** for multi-agent orchestration
- **ChromaDB** for vector storage and retrieval
- **Groq** for high-performance LLM inference
- **Sentence Transformers** for semantic embeddings
- **Streamlit** for rapid web interface development

---

*Built with ❤️ for advancing AI in governance and legal technology*