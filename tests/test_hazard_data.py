import numpy as np

from horizonguard.agents import make_agent
from horizonguard.envs import make_env
from horizonguard.fork import run_episode
from horizonguard.hazard.data import build_step_data
from horizonguard.models.softmax import SoftmaxRegression
from horizonguard.spec import State


def runs(n=6):
    out = []
    for seed in range(n):
        agent = "delta-react" if seed % 2 else "scripted"
        out.append(run_episode(make_env("research"), make_agent(agent), seed=seed))
    return out


def test_step_data_rows_and_censoring():
    trajs = runs()
    data = build_step_data(trajs)
    assert len(data) == sum(len(t) for t in trajs)
    last_rows = [np.flatnonzero(data.run == i)[-1] for i in range(len(trajs))]
    assert all(data.next_state[i] == -1 for i in last_rows)
    # violated is absorbing in the labels
    for i in range(len(data) - 1):
        if data.state[i] == State.VIOLATED and data.run[i] == data.run[i + 1]:
            assert data.state[i + 1] == State.VIOLATED


def test_transitions_from_source():
    data = build_step_data(runs())
    x, y = data.transitions_from(State.ALIGNED)
    assert len(x) == len(y) > 0
    assert set(np.unique(y)) <= {0, 1, 2}


def test_softmax_recovers_classes_and_keeps_absent_columns():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(600, 2))
    y = np.where(x[:, 0] > 0.5, 1, 0)
    model = SoftmaxRegression(n_classes=3, l2=0.1).fit(x, y)
    p = model.predict_proba(x)
    assert p.shape == (600, 3)
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    assert (p.argmax(axis=1) == y).mean() > 0.95
    assert p[:, 2].max() < 0.05
