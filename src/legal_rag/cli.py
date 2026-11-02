"""CLI: ingest official BGB or synthetic demo, ask, or benchmark retrieval."""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path

from legal_rag.answer_evaluation import evaluate_answers
from legal_rag.chunking import chunk_section
from legal_rag.config import Settings
from legal_rag.evaluation import run_benchmark
from legal_rag.faithfulness import prepare_review, score_review
from legal_rag.ollama import Ollama
from legal_rag.service import RAGService
from legal_rag.sources import BGB_XML_URL, fetch_bgb_xml, load_demo, parse_bgb_xml
from legal_rag.storage import Index


def main() -> None:
    parser = argparse.ArgumentParser(description="Local-first bilingual legal RAG")
    sub = parser.add_subparsers(dest="action", required=True)

    ingest = sub.add_parser("ingest", help="Build/rebuild local retrieval index")
    ingest.add_argument("--source", choices=["demo", "bgb"], default="demo")
    ingest.add_argument("--limit", type=int, default=None, help="Maximum BGB sections")
    ingest.add_argument("--demo-file", type=Path, default=Path("examples/demo_policies.json"))
    ingest.add_argument("--input-xml", type=Path, default=None,
                        help="Use a locally saved BGB XML file instead of downloading")
    ingest.add_argument("--dry-run", action="store_true",
                        help="Preview corpus without changing the local index")

    ask = sub.add_parser("ask", help="Ask a question in German or English")
    ask.add_argument("query", type=str)
    ask.add_argument(
        "--mode", choices=["bm25", "dense", "hybrid", "hybrid_dense3", "dense_neighbors"], default="dense"
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
        "--mode", choices=["bm25", "dense", "hybrid", "hybrid_dense3", "dense_neighbors"], default="dense"
    )

    audit_export = sub.add_parser(
        "audit-export", help="Export cited claims for independent human source review"
    )
    audit_export.add_argument(
        "--report", type=Path, default=Path("reports/bgb_answers/answer_report.json")
    )
    audit_export.add_argument(
        "--output", type=Path, default=Path("reports/faithfulness/review.json")
    )

    audit_score = sub.add_parser(
        "audit-score", help="Validate human judgments and calculate reviewed claim support"
    )
    audit_score.add_argument(
        "--annotations", type=Path, default=Path("reports/faithfulness/review.json")
    )
    audit_score.add_argument(
        "--output", type=Path, default=Path("reports/faithfulness/score")
    )

    args = parser.parse_args()
    if args.action == "audit-export":
        result = prepare_review(args.report, args.output)
        print(json.dumps({"claims_to_review": len(result["claims"]),
                          "output": str(args.output)}, indent=2))
        return
    if args.action == "audit-score":
        print(json.dumps(score_review(args.annotations, args.output), indent=2))
        return
    settings = Settings.from_env()
    if args.action == "ingest":
        if args.limit is not None and args.limit < 1:
            parser.error("--limit must be >= 1")
        if args.source != "bgb" and (args.input_xml is not None or args.limit is not None):
            parser.error("--input-xml and --limit are only valid for --source bgb")
        source_metadata = {"source_kind": args.source}
        if args.source == "demo":
            sections = load_demo(args.demo_file)
        else:
            xml = args.input_xml.read_bytes() if args.input_xml else fetch_bgb_xml()
            if len(xml) > 45 * 1024 * 1024:
                raise ValueError("BGB XML exceeds allowed uncompressed size")
            sections = parse_bgb_xml(xml, limit=args.limit)
            source_metadata.update({
                "source_location": str(args.input_xml) if args.input_xml else BGB_XML_URL,
                "source_sha256": sha256(xml).hexdigest(),
            })
        if args.dry_run:
            print(json.dumps({
                **source_metadata,
                "section_count": len(sections),
                "passage_count": sum(len(chunk_section(s)) for s in sections),
                "first_sections": [s.title for s in sections[:5]],
                "last_sections": [s.title for s in sections[-5:]],
                "index_modified": False,
            }, ensure_ascii=False, indent=2))
            return
        ollama = Ollama(settings.ollama_base_url, settings.embed_model, settings.llm_model)
        manifest = Index(settings, ollama).build(sections, source_metadata=source_metadata)
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
