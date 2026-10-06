"""The LLM prompts are part of the response-cache keys; these pins catch accidental edits."""

import json

import pytest

from langsensor import paths
from langsensor.predictors.incontext import InContextCalibration
from langsensor.predictors.proposer import FEW_SHOT, USER_TEMPLATE, prompt_signature
from langsensor.vlmap.language import FEW_SHOT_CL, USER_TEMPLATE_CL


def test_static_proposer_signature():
    assert prompt_signature() == "1bcddb96696ac026"


def test_closed_loop_prompt_differs_by_one_instruction():
    added = set(USER_TEMPLATE_CL.splitlines()) - set(USER_TEMPLATE.splitlines())
    assert len(added) == 1 and "relational sparsity" in added.pop()
    assert len(FEW_SHOT_CL) == len(FEW_SHOT) + 2


@pytest.mark.parametrize("name, signature", [
    ("llm_gpt-5.2.json", "7db466d4448b2519"),
    ("llm_gpt-5.2_ambig_val_seen.json", "805c265baba588ec"),
])
def test_incontext_signatures(name, signature):
    path = paths.STATIC_CACHE / "incontext" / name
    if not path.exists():
        pytest.skip("artifacts not downloaded")
    assert InContextCalibration(**json.loads(path.read_text())).signature() == signature
