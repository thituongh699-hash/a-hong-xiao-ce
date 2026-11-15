import json

import pytest

from src.optimization import ParameterOptimizer


def test_grid_deduplicates_and_ranks_both_directions():
    optimizer = ParameterOptimizer()
    space = {"a": [1, 1, 2], "b": [0, 1]}
    result = optimizer.run(space, lambda params: {"score": params["a"] * 10 + params["b"]})
    assert result.metadata["total_candidates"] == 4
    assert [record["score"] for record in result.ranked] == [21, 20, 11, 10]
    assert result.best["params"] == {"a": 2, "b": 1}
    low = optimizer.run(space, lambda params: {"score": params["a"] * 10 + params["b"]}, maximize=False)
    assert low.best["score"] == 10


def test_explicit_configurations_dedupe_and_seeded_bounded_selection():
    space = [{"x": i} for i in range(20)] + [{"x": 3}]
    calls = []
    def evaluate(params):
        calls.append(params["x"])
        return {"score": params["x"]}
    first = ParameterOptimizer().run(space, evaluate, max_trials=5, seed=17)
    second = ParameterOptimizer().run(space, evaluate, max_trials=5, seed=17)
    assert first.metadata["total_candidates"] == 20
    assert len(first.records) == 5
    assert [r["params"] for r in first.records] == [r["params"] for r in second.records]
    assert len(set(calls[:5])) == 5
    with pytest.raises(ValueError, match="exceeds"):
        ParameterOptimizer().run({"a": list(range(101)), "b": list(range(1000))}, evaluate, max_trials=1)


def test_errors_nonfinite_constraints_and_eligibility_never_rank():
    calls = []
    def evaluate(params):
        calls.append(params["x"])
        if params["x"] == 2:
            raise RuntimeError("bad trial")
        return {"score": float("nan") if params["x"] == 3 else params["x"], "risk": params["x"]}
    result = ParameterOptimizer().run(
        {"x": [1, 2, 3, 4, 5]}, evaluate,
        constraint=lambda params: params["x"] != 1,
        eligible=lambda metrics: metrics["risk"] < 5,
    )
    assert calls == [2, 3, 4, 5]
    assert [record["status"] for record in result.records] == ["ineligible", "error", "ineligible", "ok", "ineligible"]
    assert [record["score"] for record in result.ranked] == [4]
    assert result.records[2]["metrics"]["score"] is None
    assert result.metadata["attempted"] == 4
    assert result.metadata["skipped"] == 1


def test_resume_reuses_completed_records_and_reports_progress(tmp_path):
    checkpoint = tmp_path / "search.jsonl"
    context = {"dataset": "v1", "period": "train-2020", "evaluator_version": "v1"}
    calls = []
    events = []
    def evaluate(params):
        calls.append(params["x"])
        if len(calls) == 2:
            raise KeyboardInterrupt("interrupted")
        return {"score": params["x"]}
    with pytest.raises(KeyboardInterrupt):
        ParameterOptimizer().run({"x": [1, 2, 3]}, evaluate, checkpoint=checkpoint, context=context,
                                 progress=lambda record, done, total: events.append((done, total)))
    assert events == [(1, 3)]
    resumed = ParameterOptimizer().run({"x": [1, 2, 3]}, evaluate, checkpoint=checkpoint, context=context,
                                       progress=lambda record, done, total: events.append((done, total)))
    assert calls == [1, 2, 2, 3]
    assert resumed.metadata["resumed"] == 1
    assert resumed.metadata["completed"] == 3
    assert resumed.best["score"] == 3
    assert events == [(1, 3), (2, 3), (3, 3)]
    assert len(checkpoint.read_text(encoding="utf-8").splitlines()) == 4


def test_checkpoint_rejects_context_change_and_truncated_line(tmp_path):
    checkpoint = tmp_path / "search.jsonl"
    space = [{"x": 1}, {"x": 2}]
    optimizer = ParameterOptimizer()
    with pytest.raises(ValueError, match="context"):
        optimizer.run(space, lambda _: {"score": 1}, checkpoint=checkpoint)
    optimizer.run(space, lambda p: {"score": p["x"]}, checkpoint=checkpoint, context={"dataset": "v1"}, evaluator_version="v1")
    with pytest.raises(ValueError, match="fingerprint"):
        optimizer.run(space, lambda p: {"score": p["x"]}, checkpoint=checkpoint, context={"dataset": "v2"}, evaluator_version="v1")
    with checkpoint.open("ab") as stream:
        stream.write(b'{"type":"trial"')
    with pytest.raises(ValueError, match="truncated"):
        optimizer.run(space, lambda p: {"score": p["x"]}, checkpoint=checkpoint, context={"dataset": "v1"}, evaluator_version="v1")


def test_checkpoint_rejects_malformed_json(tmp_path):
    checkpoint = tmp_path / "search.jsonl"
    space = [{"x": 1}]
    optimizer = ParameterOptimizer()
    optimizer.run(space, lambda _: {"score": 1}, checkpoint=checkpoint, context={"dataset": "v1"}, evaluator_version="v1")
    with checkpoint.open("ab") as stream:
        stream.write(b'{"type":"trial","record":{"score":NaN}}\n')
    with pytest.raises(ValueError, match="malformed"):
        optimizer.run(space, lambda _: {"score": 1}, checkpoint=checkpoint, context={"dataset": "v1"}, evaluator_version="v1")


def test_checkpoint_requires_callback_versions_and_rejects_changed_versions(tmp_path):
    checkpoint = tmp_path / "search.jsonl"
    space = [{"x": 1}, {"x": 2}]
    evaluate = lambda p: {"score": p["x"]}
    eligible = lambda m: m["score"] > 0
    constraint = lambda p: p["x"] > 0
    optimizer = ParameterOptimizer()
    with pytest.raises(ValueError, match="evaluator_version"):
        optimizer.run(space, evaluate, checkpoint=checkpoint, context={"dataset": "v1"})
    with pytest.raises(ValueError, match="eligibility_version"):
        optimizer.run(space, evaluate, checkpoint=checkpoint, context={"dataset": "v1"},
                      evaluator_version="eval-v1", eligible=eligible)
    with pytest.raises(ValueError, match="constraint_version"):
        optimizer.run(space, evaluate, checkpoint=checkpoint, context={"dataset": "v1"},
                      evaluator_version="eval-v1", eligible=eligible, eligibility_version="eligible-v1",
                      constraint=constraint)
    kwargs = dict(checkpoint=checkpoint, context={"dataset": "v1"}, evaluator_version="eval-v1",
                  eligible=eligible, eligibility_version="eligible-v1",
                  constraint=constraint, constraint_version="constraint-v1")
    first = optimizer.run(space, evaluate, **kwargs)
    assert first.metadata["resumed"] == 0
    assert optimizer.run(space, evaluate, **kwargs).metadata["resumed"] == 2
    for field in ("evaluator_version", "eligibility_version", "constraint_version"):
        changed = {**kwargs, field: kwargs[field] + "-changed"}
        with pytest.raises(ValueError, match="fingerprint"):
            optimizer.run(space, evaluate, **changed)


def test_evaluator_output_must_be_json_serializable():
    result = ParameterOptimizer().run([{"x": 1}, {"x": 2}],
                                      lambda params: {"score": params["x"], "bad": object()}
                                      if params["x"] == 1 else {"score": 2})
    assert result.records[0]["status"] == "error"
    assert result.best["params"] == {"x": 2}
    json.dumps(result.records, allow_nan=False)


def test_callbacks_cannot_mutate_nested_search_parameters():
    space = [{"nested": {"x": 1}}]
    def evaluate(params):
        params["nested"]["x"] = 99
        return {"score": 10 ** 400}
    result = ParameterOptimizer().run(space, evaluate)
    assert result.best["score"] == 10 ** 400
    assert result.best["params"] == {"nested": {"x": 1}}
    assert space == [{"nested": {"x": 1}}]
