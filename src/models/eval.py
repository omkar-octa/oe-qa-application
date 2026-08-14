from typing import Literal

from pydantic import BaseModel, Field


class EvalQuestion(BaseModel):
    """One question from the eval bank, tests/fixtures/eval_questions.json,
    hand-maintained directly rather than generated from anything else.

    output is the hand-written reference answer, including its own file/page
    citations -- often by a short name from the question bank's own
    "short_names" mapping (e.g. "Burlinson p1") rather than the real file
    name, since that is how the question bank itself writes citations.
    """

    id: str
    section: str
    tags: list[str] = Field(default_factory=list)
    input: str
    output: str


class EvalQuestionBank(BaseModel):
    """The full contents of tests/fixtures/eval_questions.json: every question
    plus the short-name -> real-file-name mapping used to write their
    citations."""

    short_names: dict[str, str] = Field(default_factory=dict)
    questions: list[EvalQuestion] = Field(default_factory=list)


class EvalResult(BaseModel):
    """One graded run of an EvalQuestion against a live QAAgent.

    verdict and reasoning come from utils.eval_judge.EvalGrader, an LLM
    judge: these answers are open-ended prose, so exact-match scoring
    against `output` is not viable.

    model, input_tokens, output_tokens and duration_seconds describe the
    QAAgent call that produced actual_answer, not the judge call that graded
    it -- the judge always runs on models.config.settings.claude_model so
    that comparing results across different answering models is comparing
    against the same grader.
    """

    question: EvalQuestion
    actual_answer: str
    verdict: Literal["pass", "partial", "fail"]
    reasoning: str
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    duration_seconds: float = 0.0
