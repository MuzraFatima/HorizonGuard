from horizonguard.bench.grid import DRIFT_TYPES, fork_grid


def test_regular_grid_stops_before_violation():
    fid = [0.9] * 40
    assert fork_grid(fid, violation_step=17, every=5, start=2) == [2, 7, 12]
    assert fork_grid(fid, None, every=10, start=0, max_points=10) == [0, 10, 20, 30]


def test_adaptive_points_are_added_where_fidelity_falls():
    fid = [0.9] * 10 + [0.7, 0.5, 0.4] + [0.4] * 10
    grid = fork_grid(fid, None, every=10, start=2)
    assert 12 in grid or 13 in grid
    assert 2 in grid


def test_cap_keeps_adaptive_points_first():
    fid = [0.9 - 0.04 * t for t in range(20)]
    grid = fork_grid(fid, None, every=2, start=2, max_points=4, adaptive_drop=0.05)
    assert len(grid) == 4


def test_drift_types_are_distinct():
    assert len(DRIFT_TYPES) == 5
    assert len({repr(d) for d in DRIFT_TYPES.values()}) == 5
