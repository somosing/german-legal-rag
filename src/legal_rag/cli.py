"""CLI: ingest official BGB or synthetic demo, ask, or benchmark retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from legal_rag.answer_evaluation import evaluate_answers
from legal_rag.config import Settings
from legal_rag.evaluation import run_benchmark
from legal_rag.ollama import Ollama
from legal_rag.service import RAGService
from legal_rag.sources import fetch_bgb_xml, load_demo, parse_bgb_xml
from legal_rag.storage import Index


def main() -> None:
    parser = argparse.ArgumentParser(description="Local-first bilingual legal RAG")
    sub = parser.add_subparsers(dest="action", required=True)

    ingest = sub.add_parser("ingest", help="Build/rebuild local retrieval index")
    ingest.add_argument("--source", choices=["demo", "bgb"], default="demo")
    ingest.add_argument("--limit", type=int, default=None, help="Maximum BGB sections")
    ingest.add_argument("--demo-file", type=Path, default=Path("examples/demo_policies.json"))

    ask = sub.add_parser("ask", help="Ask a question in German or English")
    ask.add_argument("query", type=str)
    ask.add_argument(
        "--mode", choices=["bm25", "dense", "hybrid", "hybrid_dense3"], default="dense"
    )

    evaluate = sub.add_parser("eval", help="Compare four retriever modes")
    evaluate.add_argument("--gold", type=Path, default=Path("examples/demo_questions.json"))
    evaluate.add_argument("--output", type=Path, default=Path("reports"))

    answer_eval = sub.add_parser(
        "eval-answers", help="Evaluate structured answer/citation contract"
    )
    answer_eval.add_argument(
        "--gold", type=Path, default=Path("examples/bgb_answer_questions.json")
    )
    answer_eval.add_argument("--output", type=Path, default=Path("reports/bgb_answers"))
    answer_eval.add_argument(
        "--mode", choices=["bm25", "dense", "hybrid", "hybrid_dense3"], default="dense"
    )

    args = parser.parse_args()
    settings = Settings.from_env()
    if args.action == "ingest":
        if args.limit is not None and args.limit < 1:
            parser.error("--limit must be >= 1")
        sections = (
            load_demo(args.demo_file)
            if args.source == "demo"
            else parse_bgb_xml(fetch_bgb_xml(), limit=args.limit)
        )
        ollama = Ollama(settings.ollama_base_url, settings.embed_model, settings.llm_model)
        manifest = Index(settings, ollama).build(sections)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    elif args.action == "ask":
        result = RAGService(settings).ask(args.query, args.mode)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.action == "eval-answers":
        service = RAGService(settings)
        result = evaluate_answers(service, args.gold, args.output, args.mode)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    else:
        service = RAGService(settings)
        result = run_benchmark(service.retriever, args.gold, args.output)
        print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
