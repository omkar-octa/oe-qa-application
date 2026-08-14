"""Runs tests/fixtures/eval_questions.json's question bank against a live
QAAgent on the Postgres/pgvector hybrid-search backend, and grades each answer
against the bank's own hand-written reference answer -- docs/roadmap.md item 10.

Costs two real Claude calls per question (QAAgent.answer(), then
EvalGrader.grade()) plus whatever search/read_source calls QAAgent's own
tool-use loop makes, and needs Postgres running with an index already written
by `python main.py index`. Questions run concurrently (--workers, default 5):
each worker thread gets its own QAAgent/PostgresSearchIndex/Postgres
connection and its own EvalGrader, since a psycopg connection is not safe to
share across threads and isolating everything per-thread avoids having to
reason about it beyond that.

tests/fixtures/eval_questions.json is hand-maintained directly (structured
`{id, section, tags, input, output}` questions plus a short-name-to-real-file-
name table); edit it in place rather than through a generated intermediate.

Known gap worth reading results in light of: PostgresSearchIndex attaches no
neighbouring-chunk context around a match unless constructed with
attach_context=True (off by default here, same as everywhere else -- see
models.config.settings.attach_search_context), so a `multi-hop` or `footnote`
question relies on QAAgent's read_source escalation tool rather than
surrounding context being handed to it automatically.

Run from src/:
    python scripts/run_evals.py
    python scripts/run_evals.py --tag figure --tag trap
    python scripts/run_evals.py --id C5 --id S4
    python scripts/run_evals.py --limit 10
    python scripts/run_evals.py --workers 8
    python scripts/run_evals.py --model claude-haiku-4-5 --model claude-sonnet-5
    python scripts/run_evals.py --summarize data/eval_results_claude-haiku-4-5.json data/eval_results.json

--model is repeatable: each answering model runs the full (filtered)
question set in turn -- grading always uses settings.claude_model, so the
judge is held constant across the comparison -- and results are written to
one output file per model plus a pass-rate/token/latency comparison table
once all models have run.

--summarize skips running anything (no Postgres, no API calls) and instead
reprints that same per-tag and model-comparison summary from already-graded
result files on disk -- useful for comparing runs made at different times,
or for re-reading a comparison without paying for it twice. Rows are grouped
by their own `model` field; a file graded before that field existed falls
back to being labelled with its own filename.
"""

import argparse
import json
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.config import settings  # noqa: E402
from models.eval import EvalQuestion, EvalQuestionBank, EvalResult  # noqa: E402
from utils.embedder import Embedder  # noqa: E402
from utils.eval_judge import EvalGrader  # noqa: E402
from utils.postgres_search_index import PostgresSearchIndex  # noqa: E402
from utils.qa_agent import QAAgent  # noqa: E402

_QUESTIONS_JSON_PATH = (
    Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "eval_questions.json"
)
_DEFAULT_OUTPUT = Path("data/eval_results.json")
_DEFAULT_WORKERS = 5

_thread_local = threading.local()


def _select_questions(questions, tags, ids, limit):
    if tags:
        wanted = set(tags)
        questions = [q for q in questions if wanted & set(q.tags)]
    if ids:
        wanted_ids = set(ids)
        questions = [q for q in questions if q.id in wanted_ids]
    if limit:
        questions = questions[:limit]
    return questions


def _agent_and_grader(short_name_map: dict[str, str], model: str) -> tuple[QAAgent, EvalGrader]:
    """One QAAgent per (thread, model) -- each with its own Postgres
    connection -- and one EvalGrader per thread, shared across models since
    grading always runs on settings.claude_model regardless of which model
    answered. Built on first use and reused for every question that lands on
    that thread thereafter."""
    if not hasattr(_thread_local, "agents"):
        _thread_local.agents = {}
        _thread_local.grader = EvalGrader(short_name_map=short_name_map)
    if model not in _thread_local.agents:
        search_index = PostgresSearchIndex(embedder=Embedder())
        _thread_local.agents[model] = QAAgent(search_index, model=model)
    return _thread_local.agents[model], _thread_local.grader


def _run_one(
    question: EvalQuestion, top_k: int, short_name_map: dict[str, str], model: str
) -> EvalResult:
    agent, grader = _agent_and_grader(short_name_map, model)
    started = time.monotonic()
    answer = agent.answer(question.input, top_k=top_k)
    duration_seconds = time.monotonic() - started
    return grader.grade(
        question,
        answer,
        model=model,
        input_tokens=agent.total_input_tokens,
        output_tokens=agent.total_output_tokens,
        duration_seconds=duration_seconds,
    )


def _sanitize_for_filename(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "-", model)


def _print_summary(results: list[EvalResult]) -> None:
    by_tag = defaultdict(list)
    for result in results:
        for tag in result.question.tags or ["(untagged)"]:
            by_tag[tag].append(result)

    print("\n=== Results by tag ===")
    for tag in sorted(by_tag):
        tagged = by_tag[tag]
        passed = sum(1 for r in tagged if r.verdict == "pass")
        print(f"  {tag}: {passed}/{len(tagged)} pass")

    overall_pass = sum(1 for r in results if r.verdict == "pass")
    overall_partial = sum(1 for r in results if r.verdict == "partial")
    overall_fail = sum(1 for r in results if r.verdict == "fail")
    print(
        f"\nOverall: {overall_pass} pass, {overall_partial} partial, "
        f"{overall_fail} fail, of {len(results)}"
    )

    failures = [r for r in results if r.verdict != "pass"]
    if failures:
        print("\n=== Not passing ===")
        for r in failures:
            tags = ", ".join(r.question.tags)
            print(f"  [{r.question.id}] ({tags}) {r.verdict.upper()}: {r.reasoning}")


def _print_model_comparison(results_by_model: dict[str, list[EvalResult]]) -> None:
    """Side-by-side accuracy/cost/speed for each --model run over the same
    question set. Only prints when there's more than one model to compare;
    with a single model this would just restate _print_summary's numbers."""
    if len(results_by_model) < 2:
        return

    print("\n=== Model comparison ===")
    header = f"  {'model':<28} {'pass':>10} {'avg in tok':>11} {'avg out tok':>12} {'avg sec':>8}"
    print(header)
    for model, results in results_by_model.items():
        n = len(results)
        passed = sum(1 for r in results if r.verdict == "pass")
        avg_in = sum(r.input_tokens for r in results) / n
        avg_out = sum(r.output_tokens for r in results) / n
        avg_sec = sum(r.duration_seconds for r in results) / n
        print(
            f"  {model:<28} {f'{passed}/{n}':>10} {avg_in:>11.0f} {avg_out:>12.0f} {avg_sec:>8.1f}"
        )


def _summarize_files(paths: list[Path]) -> None:
    """Reload already-graded result files and print the same per-tag and
    model-comparison summaries a live run ends with, without touching
    Postgres or the API. Grouped by each result's own `model` field so files
    that mix models (or that came from a single-model run) still land in the
    right group; a result graded before that field existed (blank model)
    falls back to its file's stem so it still shows up under its own label."""
    results_by_model: dict[str, list[EvalResult]] = defaultdict(list)
    for path in paths:
        if not path.exists():
            print(f"{path} not found.")
            sys.exit(1)
        rows = json.loads(path.read_text(encoding="utf-8"))
        for row in rows:
            result = EvalResult.model_validate(row)
            results_by_model[result.model or path.stem].append(result)

    for model, results in results_by_model.items():
        print(f"\n=== {model} ({len(results)} results) ===")
        _print_summary(results)

    _print_model_comparison(results_by_model)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Run tests/fixtures/eval_questions.json against QAAgent (Postgres backend) and grade it."
    )
    parser.add_argument(
        "--tag", action="append", default=[], help="Only run questions carrying this tag; repeatable."
    )
    parser.add_argument(
        "--id", action="append", default=[], dest="ids", help="Only run this question ID; repeatable."
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Cap the number of questions run, after tag/id filtering."
    )
    parser.add_argument("--top-k", type=int, default=settings.top_k)
    parser.add_argument(
        "--workers", type=int, default=_DEFAULT_WORKERS, help="Questions to run concurrently."
    )
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        default=[],
        help=(
            "Answering model to run the question set against (grading always uses "
            "settings.claude_model); repeatable to run the same questions through "
            "several models for a side-by-side comparison. Defaults to settings.claude_model."
        ),
    )
    parser.add_argument(
        "--summarize",
        nargs="+",
        type=Path,
        metavar="FILE",
        help=(
            "Skip running anything and just print the per-tag and model-comparison "
            "summary for these already-graded result files (e.g. from earlier --model "
            "runs). All other arguments are ignored."
        ),
    )
    args = parser.parse_args()

    if args.summarize:
        _summarize_files(args.summarize)
        return

    models = args.models or [settings.claude_model]

    if not _QUESTIONS_JSON_PATH.exists():
        print(f"{_QUESTIONS_JSON_PATH} not found.")
        sys.exit(1)

    bank = EvalQuestionBank.model_validate_json(_QUESTIONS_JSON_PATH.read_text(encoding="utf-8"))
    questions = _select_questions(bank.questions, args.tag, args.ids, args.limit)
    if not questions:
        print("No questions matched the given filters.")
        sys.exit(1)

    results_by_model: dict[str, list[EvalResult]] = {}
    for model in models:
        if len(models) > 1:
            print(f"\n=== Running against {model} ===")

        results: list[EvalResult | None] = [None] * len(questions)
        completed = 0
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_index = {
                executor.submit(_run_one, question, args.top_k, bank.short_names, model): i
                for i, question in enumerate(questions)
            }
            for future in as_completed(future_to_index):
                i = future_to_index[future]
                result = future.result()
                results[i] = result
                completed += 1
                tags = ", ".join(result.question.tags)
                print(
                    f"[{completed}/{len(questions)}] {result.question.id} ({tags}) "
                    f"-> {result.verdict.upper()}: {result.reasoning}"
                )

        output_path = (
            args.output
            if len(models) == 1
            else args.output.with_name(
                f"{args.output.stem}_{_sanitize_for_filename(model)}{args.output.suffix}"
            )
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps([r.model_dump() for r in results], indent=2), encoding="utf-8")
        print(f"\nWrote {len(results)} graded results to {output_path}")

        _print_summary(results)
        results_by_model[model] = results

    _print_model_comparison(results_by_model)


if __name__ == "__main__":
    main()
