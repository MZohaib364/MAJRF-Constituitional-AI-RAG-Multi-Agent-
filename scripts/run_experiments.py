from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import EvaluationResult, compute_metrics
from src.orchestrator import load_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run compliance experiments and compute metrics.")
    parser.add_argument("--input", type=str, default="Data/samples", help="Directory with policy test cases (.txt).")
    parser.add_argument("--labels", type=str, default="Data/samples/labels.json", help="Ground-truth label file.")
    parser.add_argument("--root", type=str, default=".", help="Project root.")
    parser.add_argument("--pipeline", type=str, choices=["sequential", "langgraph"], default="langgraph")
    parser.add_argument("--use-llm", action="store_true", help="Enable Groq LLM reasoning.")
    parser.add_argument("--groq-model", type=str, default="llama-3.3-70b-versatile")
    parser.add_argument("--output", type=str, default="experiments/results", help="Directory to store experiment logs.")
    return parser.parse_args()


def gather_predictions(pipeline, input_path: Path) -> Dict[str, List[str]]:
    predictions: Dict[str, List[str]] = {}
    for file_path in sorted(input_path.glob("*.txt")):
        state = pipeline.evaluate(file_path)
        articles = [conflict.article_id for conflict in state.conflicts]
        if not articles and state.llm_analysis:
            articles = [str(a) for a in state.llm_analysis.get("articles_involved", [])]
        predictions[file_path.stem] = articles
    return predictions


def save_report(
    output_dir: Path,
    predictions: Dict[str, List[str]],
    metrics: EvaluationResult,
    args: argparse.Namespace,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow()
    report = {
        "timestamp": timestamp.isoformat(),
        "pipeline": args.pipeline,
        "use_llm": args.use_llm,
        "groq_model": args.groq_model,
        "predictions": predictions,
        "metrics": metrics.__dict__,
    }
    path = output_dir / f"metrics_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    labels_path = Path(args.labels).resolve()
    pipeline = load_pipeline(
        root_dir=args.root,
        engine=args.pipeline,
        use_llm=args.use_llm,
        llm_model=args.groq_model,
    )
    predictions = gather_predictions(pipeline, input_path)
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    metrics = compute_metrics(predictions, labels)
    print(f"Precision: {metrics.precision:.2f} | Recall: {metrics.recall:.2f} | F1: {metrics.f1:.2f} | Coverage: {metrics.coverage:.2f}")
    report_path = save_report(Path(args.output), predictions, metrics, args)
    print(f"Saved experiment report to {report_path}")


if __name__ == "__main__":
    main()

