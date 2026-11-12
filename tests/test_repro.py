import numpy as np

from horizonguard.repro import config_hash, load_config, run_record, seed_everything


def test_seed_everything_is_deterministic():
    a = seed_everything(7).normal(size=5)
    b = seed_everything(7).normal(size=5)
    np.testing.assert_array_equal(a, b)


def test_config_hash_ignores_key_order():
    assert config_hash({"a": 1, "b": [1, 2]}) == config_hash({"b": [1, 2], "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_load_default_config_and_record(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("seed: 3\nname: demo\n", encoding="utf-8")
    cfg = load_config(path)
    record = run_record(cfg, seed=cfg["seed"], stage="unit")
    assert record["seed"] == 3
    assert record["stage"] == "unit"
    assert len(record["config_hash"]) == 12
