from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from src.orchestrator import load_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Constitutional Compliance Checker")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to a policy file or directory containing policy cases.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs",
        help="Directory where evaluation results will be stored.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Project root containing the Data directory.",
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        choices=["sequential", "langgraph"],
        default="sequential",
        help="Execution engine: simple sequential pipeline or LangGraph multi-agent graph.",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Enable Groq LLM reasoning agent (requires GROQ_API_KEY).",
    )
    parser.add_argument(
        "--groq-model",
        type=str,
        default="llama-3.3-70b-versatile",
        help="Groq model name to use when --use-llm is supplied.",
    )
    return parser.parse_args()


def evaluate_cases(input_path: Path, pipeline, output_dir: Path) -> List[Path]:
    outputs: List[Path] = []
    if input_path.is_file():
        result_path = _evaluate_single(input_path, pipeline, output_dir)
        outputs.append(result_path)
    else:
        for file_path in input_path.glob("*.txt"):
            outputs.append(_evaluate_single(file_path, pipeline, output_dir))
    return outputs


def _evaluate_single(file_path: Path, pipeline, output_dir: Path) -> Path:
    state = pipeline.evaluate(file_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{state.policy_id}.json"
    output_path.write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    pipeline = load_pipeline(
        root_dir=args.root,
        engine=args.pipeline,
        use_llm=args.use_llm,
        llm_model=args.groq_model,
    )
    results = evaluate_cases(input_path, pipeline, output_dir)
    for path in results:
        print(f"Saved evaluation: {path}")


if __name__ == "__main__":
    main()

