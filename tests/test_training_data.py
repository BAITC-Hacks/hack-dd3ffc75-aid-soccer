from copy import deepcopy
import json

import pytest

from llm_advisor import PROMPT_VERSION, context_hash
from tools.export_training_data import build_training_split


def examples(count=15):
    experiences, reviews = [], []
    for index in range(count):
        candidate = {"candidate_id": f"a{index}|MID|b", "cell_id": f"a{index}|MID",
                     "served_arpu": 1000 + index}
        context = dict(prompt_version=PROMPT_VERSION, max_recommendations=2,
                       candidates=[candidate], tariffs=[], channels={}, resources={"remaining_budget": 100})
        key = context_hash(context)
        experiences.append(dict(context_hash=key, context=context, proposal=None))
        reviews.append(dict(context_hash=key, approved=True, reviewer="test analyst",
                            candidate_ids=[candidate["candidate_id"]], reason="Reviewed fixture choice"))
    return experiences, reviews


def test_exports_jsonl_shape_and_disjoint_scenario_groups():
    experiences, reviews = examples()
    train, validation, manifest = build_training_split(experiences, reviews)
    assert len(train) == 12 and len(validation) == 3
    training_groups = {r["scenario_group"] for r in manifest if r["split"] == "train"}
    validation_groups = {r["scenario_group"] for r in manifest if r["split"] == "validation"}
    assert not training_groups & validation_groups
    assert [m["role"] for m in train[0]["messages"]] == ["system", "user", "assistant"]
    assert json.loads(train[0]["messages"][2]["content"])["candidate_ids"]
    assert build_training_split(list(reversed(experiences)), list(reversed(reviews))) == (train, validation, manifest)


def test_same_context_deduplicated_and_resource_variants_stay_in_same_split():
    experiences, reviews = examples()
    duplicate = deepcopy(experiences[0])
    duplicate["context"]["resources"]["remaining_budget"] = 200
    duplicate["context_hash"] = context_hash(duplicate["context"])
    experiences += [experiences[0], duplicate]
    reviews.append({**reviews[0], "context_hash": duplicate["context_hash"]})
    _, _, manifest = build_training_split(experiences, reviews)
    assert len(manifest) == 16
    related = [r for r in manifest if r["context_hash"] in {experiences[0]["context_hash"], duplicate["context_hash"]}]
    assert len({r["scenario_group"] for r in related}) == 1
    assert len({r["split"] for r in related}) == 1


def test_unreviewed_model_proposals_never_become_training_labels():
    experiences, reviews = examples()
    for review in reviews:
        review["approved"] = False
    with pytest.raises(ValueError, match="independent scenario"):
        build_training_split(experiences, reviews)


def test_repeated_seeds_not_independent_training_examples():
    experiences, reviews = examples(1)
    with pytest.raises(ValueError, match="independent scenario"):
        build_training_split(experiences * 20, reviews * 20)


@pytest.mark.parametrize("mutation,match", [
    (lambda es, rs: rs[0].update(reviewer=""), "reviewer"),
    (lambda es, rs: es[0].update(context_hash="wrong"), "hash"),
    (lambda es, rs: rs.append({**rs[0], "candidate_ids": []}), "Conflicting"),
])
def test_invalid_reviews_are_rejected(mutation, match):
    experiences, reviews = examples()
    mutation(experiences, reviews)
    with pytest.raises(ValueError, match=match):
        build_training_split(experiences, reviews)


def test_too_few_training_examples_are_rejected():
    experiences, reviews = examples(10)
    with pytest.raises(ValueError, match="At least 10"):
        build_training_split(experiences, reviews)
