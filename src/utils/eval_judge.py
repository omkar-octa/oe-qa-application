"""Grades one QAAgent answer against tests/fixtures/eval_questions.json's
hand-written reference answer, one Claude call per question.

These are open-ended prose answers with their own citation style (often a
short name like "Burlinson p1" rather than the real file name), so exact-match
scoring against the reference is not viable -- an LLM judge reads both and
decides whether the content and the citation are right.
"""

import re
from pathlib import Path

import anthropic

from models.config import settings
from models.eval import EvalQuestion, EvalResult

_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "eval_judge.prompt"

# A verdict plus one or two sentences of reasoning; generous relative to what
# that actually needs, for the same reason MetadataEnhancer.SUMMARY_MAX_TOKENS
# is: adaptive thinking's own tokens count against this budget too, and on a
# question whose reference and actual answer are both long (a multi-hop or
# cross-doc question especially) thinking alone measurably ran the budget out
# before any verdict was written -- see the explicit stop_reason check below.
GRADE_MAX_TOKENS = 1024

_VERDICT_LINE = re.compile(r"VERDICT:\s*(pass|partial|fail)", re.IGNORECASE)
_REASONING_LINE = re.compile(r"REASONING:\s*(.*)", re.IGNORECASE | re.DOTALL)


class EvalGrader:
    def __init__(self, client: anthropic.Anthropic | None = None, short_name_map: dict[str, str] | None = None):
        self._client = client or anthropic.Anthropic(api_key=settings.claude_api_key)
        self._system_prompt = _PROMPT_PATH.read_text(encoding="utf-8")
        self._short_name_map = short_name_map or {}

    def grade(
        self,
        question: EvalQuestion,
        actual_answer: str,
        model: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        duration_seconds: float = 0.0,
    ) -> EvalResult:
        """model/input_tokens/output_tokens/duration_seconds describe the
        QAAgent call that produced actual_answer -- passed through so the
        result carries them, not used in grading itself, which always runs
        on settings.claude_model regardless of which model answered."""
        response = self._client.messages.create(
            model=settings.claude_model,
            max_tokens=GRADE_MAX_TOKENS,
            system=self._system_prompt,
            messages=[{"role": "user", "content": self._build_content(question, actual_answer)}],
        )
        metadata = {
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "duration_seconds": duration_seconds,
        }

        if response.stop_reason == "refusal":
            return EvalResult(
                question=question,
                actual_answer=actual_answer,
                verdict="fail",
                reasoning="Judge refused to grade this answer.",
                **metadata,
            )

        # Adaptive thinking's own tokens count against max_tokens, and can
        # consume the whole budget before any visible text is written --
        # exactly the failure metadata_enhancer.summarize() guards against.
        # Left unchecked this produces a blank reasoning string that reads as
        # a genuine (if terse) fail rather than the ungraded question it is.
        if response.stop_reason == "max_tokens":
            return EvalResult(
                question=question,
                actual_answer=actual_answer,
                verdict="fail",
                reasoning="Judge response was truncated (max_tokens) before a verdict could be written; re-run this question.",
                **metadata,
            )

        text = "".join(block.text for block in response.content if block.type == "text").strip()
        verdict, reasoning = self._parse_verdict(text)
        return EvalResult(
            question=question,
            actual_answer=actual_answer,
            verdict=verdict,
            reasoning=reasoning,
            **metadata,
        )

    def _build_content(self, question: EvalQuestion, actual_answer: str) -> str:
        parts = [
            f"Question ID: {question.id}",
            f"Tags: {', '.join(question.tags) or '(none)'}",
            f"Question: {question.input}",
            f"Reference answer: {question.output}",
        ]
        if self._short_name_map:
            mapping = "\n".join(
                f"- {short} -> {file}" for short, file in self._short_name_map.items()
            )
            parts.append(f"Short name -> file mapping:\n{mapping}")
        parts.append(f"System's actual answer: {actual_answer}")
        return "\n\n".join(parts)

    @staticmethod
    def _parse_verdict(text: str) -> tuple[str, str]:
        # A malformed response (wrong format, or a max_tokens cut-off before
        # VERDICT: was written) defaults to "fail" rather than "pass": a
        # question that could not be graded should get a human's attention,
        # not a silent free pass.
        verdict_match = _VERDICT_LINE.search(text)
        verdict = verdict_match.group(1).lower() if verdict_match else "fail"

        reasoning_match = _REASONING_LINE.search(text)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else text

        return verdict, reasoning
