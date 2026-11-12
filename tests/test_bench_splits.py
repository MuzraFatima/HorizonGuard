import itertools

import pytest

from horizonguard.bench.splits import (
    default_split,
    leave_one_out,
    load_splits,
    runs_table,
    save_splits,
)


def fake_records():
    recs = []
    for fam, scaf, env, drift, rep in itertools.product(
        ["atlas", "birch", "ember"],
        ["react", "plan"],
        ["portfolio", "codebase"],
        ["nudge", "context"],
        range(3),
    ):
        run = f"{env}.{fam}-{scaf}.{drift}.{rep}"
        for t in (2, 7, 12):
            recs.append(
                {
                    "run_id": run,
                    "agent": f"{fam}-{scaf}",
                    "family": fam,
                    "scaffold": scaf,
                    "env": env,
                    "template": "default",
                    "drift_type": drift,
                    "t": t,
                }
            )
    return recs


def test_default_split_keeps_runs_whole_and_holds_out_ood():
    recs = fake_records()
    split = default_split(recs)
    parts = [split.train, split.calibration, split.test, split.ood_test]
    flat = [rid for p in parts for rid in p]
    assert len(flat) == len(set(flat)) == len(runs_table(recs))
    table = runs_table(recs)
    for rid in split.train + split.calibration + split.test:
        assert table[rid]["family"] != "ember" and table[rid]["drift_type"] != "context"
    for r in split.select(recs, "train"):
        assert split.part_of(r["run_id"]) == "train"


def test_split_is_deterministic():
    recs = fake_records()
    assert default_split(recs, seed=3).train == default_split(recs, seed=3).train
    assert default_split(recs, seed=3).train != default_split(recs, seed=4).train


def test_leave_one_out_covers_every_value():
    recs = fake_records()
    splits = list(leave_one_out(recs, "family"))
    assert [s.name for s in splits] == ["loo-family-atlas", "loo-family-birch", "loo-family-ember"]
    table = runs_table(recs)
    for s in splits:
        value = s.held_out["family"][0]
        assert all(table[r]["family"] == value for r in s.ood_test)
        assert all(table[r]["family"] != value for r in s.train + s.calibration)
    with pytest.raises(ValueError):
        list(leave_one_out(recs, "colour"))


def test_save_and_load(tmp_path):
    splits = [default_split(fake_records()), *leave_one_out(fake_records(), "env")]
    save_splits(splits, tmp_path / "splits.json")
    back = load_splits(tmp_path / "splits.json")
    assert back["default"].train == splits[0].train
    assert set(back) == {s.name for s in splits}
