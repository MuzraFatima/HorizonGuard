import pytest

from horizonguard.logging import StepRecord, Trajectory, group_runs, read_jsonl, write_jsonl


def record(run="r1", t=0, violated=False):
    return StepRecord(
        run_id=run,
        env="portfolio",
        agent="atlas-react",
        family="atlas",
        scaffold="react",
        seed=1,
        t=t,
        action="rebalance",
        action_kind="goal",
        observation="prices moved",
        reasoning="stick to the mandate",
        tool="trade",
        clock=float(t),
        fidelity=0.9,
        violated=violated,
    )


@pytest.mark.parametrize("name", ["runs.jsonl", "runs.jsonl.gz"])
def test_round_trip(tmp_path, name):
    recs = [record(t=0), record(t=1, violated=True), record(run="r2")]
    path = tmp_path / name
    assert write_jsonl(recs, path) == 3
    back = list(read_jsonl(path))
    assert back == recs


def test_group_runs_and_trajectory_properties():
    runs = group_runs([record(t=1, violated=True), record(t=0), record(run="r2")])
    first = next(r for r in runs if r.run_id == "r1")
    assert [s.t for s in first.steps] == [0, 1]
    assert first.violation_step == 1 and first.violated
    assert first.fidelity() == [0.9, 0.9]
    assert not Trajectory([record(run="r2")]).violated


def test_observable_view_hides_labels():
    view = record().observable()
    assert "fidelity" not in view and "action_kind" not in view and "violated" not in view
    assert view["action"] == "rebalance"


def test_unknown_version_rejected():
    data = record().__dict__ | {"schema_version": 99}
    with pytest.raises(ValueError):
        StepRecord.from_dict(data)
