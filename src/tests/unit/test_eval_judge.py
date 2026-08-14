from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from models.eval import EvalQuestion
from utils.eval_judge import EvalGrader


def make_question(**overrides) -> EvalQuestion:
    defaults = dict(
        id="B1",
        section="Burlinson",
        tags=["single"],
        input="Which three low-carbon technologies does the study track?",
        output="Solar PV, solar water heating and EVs. Burlinson p1.",
    )
    defaults.update(overrides)
    return EvalQuestion(**defaults)


def text_response(text: str, stop_reason: str = "end_turn") -> SimpleNamespace:
    return SimpleNamespace(
        stop_reason=stop_reason, content=[SimpleNamespace(type="text", text=text)]
    )


@pytest.mark.unit
def test_grade_parses_verdict_and_reasoning():
    client = MagicMock()
    client.messages.create.return_value = text_response(
        "VERDICT: pass\nREASONING: Content and citation both match the reference."
    )

    result = EvalGrader(client=client).grade(make_question(), "Solar PV, solar water heating and EVs (p1).")

    assert result.verdict == "pass"
    assert result.reasoning == "Content and citation both match the reference."
    assert result.question.id == "B1"
    assert result.actual_answer == "Solar PV, solar water heating and EVs (p1)."


@pytest.mark.unit
def test_grade_parses_partial_verdict():
    client = MagicMock()
    client.messages.create.return_value = text_response(
        "VERDICT: partial\nREASONING: Content is right but no page citation is given."
    )

    result = EvalGrader(client=client).grade(make_question(), "Solar PV, solar water heating and EVs.")

    assert result.verdict == "partial"


@pytest.mark.unit
def test_grade_is_case_insensitive_on_verdict():
    client = MagicMock()
    client.messages.create.return_value = text_response("Verdict: Fail\nReasoning: Wrong technologies named.")

    result = EvalGrader(client=client).grade(make_question(), "Wind and nuclear.")

    assert result.verdict == "fail"


@pytest.mark.unit
def test_grade_defaults_to_fail_on_malformed_response():
    # A judge response with no VERDICT: line -- whether from a format slip or
    # a truncated answer -- must not be silently treated as a pass.
    client = MagicMock()
    client.messages.create.return_value = text_response("This answer looks broadly correct to me.")

    result = EvalGrader(client=client).grade(make_question(), "Solar PV, solar water heating and EVs.")

    assert result.verdict == "fail"
    assert result.reasoning == "This answer looks broadly correct to me."


@pytest.mark.unit
def test_grade_handles_max_tokens_truncation_as_fail_not_blank():
    # Adaptive thinking can consume the whole budget before any visible text
    # is written, leaving response.content with no text blocks at all. That
    # must surface as a clearly-flagged truncation, not a fail with an empty
    # reasoning string that reads as a genuine (if terse) verdict.
    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(stop_reason="max_tokens", content=[])

    result = EvalGrader(client=client).grade(make_question(), "Some answer.")

    assert result.verdict == "fail"
    assert "truncated" in result.reasoning.lower()


@pytest.mark.unit
def test_grade_handles_refusal_as_fail():
    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(stop_reason="refusal", content=[])

    result = EvalGrader(client=client).grade(make_question(), "Some answer.")

    assert result.verdict == "fail"
    assert "refused" in result.reasoning.lower()


@pytest.mark.unit
def test_grade_includes_short_name_mapping_in_the_prompt():
    client = MagicMock()
    client.messages.create.return_value = text_response("VERDICT: pass\nREASONING: fine.")

    EvalGrader(client=client, short_name_map={"Burlinson": "1-s2.0-S0140988325000672-main.pdf"}).grade(
        make_question(), "Some answer."
    )

    content = client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "Burlinson -> 1-s2.0-S0140988325000672-main.pdf" in content


@pytest.mark.unit
def test_grade_omits_mapping_section_when_none_given():
    client = MagicMock()
    client.messages.create.return_value = text_response("VERDICT: pass\nREASONING: fine.")

    EvalGrader(client=client).grade(make_question(), "Some answer.")

    content = client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "Short name" not in content


@pytest.mark.unit
def test_grade_includes_question_and_reference_and_actual_answer():
    client = MagicMock()
    client.messages.create.return_value = text_response("VERDICT: pass\nREASONING: fine.")

    EvalGrader(client=client).grade(make_question(), "My actual answer text.")

    content = client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "Which three low-carbon technologies" in content
    assert "Solar PV, solar water heating and EVs. Burlinson p1." in content
    assert "My actual answer text." in content
