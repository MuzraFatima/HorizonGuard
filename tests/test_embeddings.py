import numpy as np
import pytest

from horizonguard.agents import make_agent
from horizonguard.embeddings import AgentNormalizer, HashingEncoder, PrefixFeaturizer
from horizonguard.envs import make_env
from horizonguard.fork import run_episode
from horizonguard.models import LogisticRegression, RidgeRegression


def test_hashing_encoder_is_deterministic_and_normalised():
    enc = HashingEncoder(dim=32)
    a = enc.encode("stay on the stated goal")
    np.testing.assert_array_equal(a, enc.encode("Stay on the STATED goal"))
    assert np.linalg.norm(a) == pytest.approx(1.0)
    assert np.linalg.norm(enc.encode("")) == 0


def test_featurizer_shapes_and_observable_only():
    traj = run_episode(make_env("research"), make_agent("birch-react"), seed=3)
    feat = PrefixFeaturizer()
    x = feat(traj.steps[:10])
    assert x.shape == (len(feat.feature_names()),)
    assert np.all(np.isfinite(x))
    # changing hidden labels must not change features
    for s in traj.steps:
        s.fidelity, s.action_kind, s.violated = 0.0, "goal", False
        s.extra["latent_drift"] = 0.0
    np.testing.assert_array_equal(feat(traj.steps[:10]), x)


def test_goal_distance_grows_for_drifting_agent():
    feat = PrefixFeaturizer()
    names = feat.feature_names()
    i = names.index("geo:dist_ema")
    drifting, steady = [], []
    for seed in range(15):
        d = run_episode(make_env("research"), make_agent("delta-react"), seed=seed)
        s = run_episode(make_env("research"), make_agent("scripted"), seed=seed)
        drifting.append(feat(d.steps[: min(15, len(d))])[i])
        steady.append(feat(s.steps[:15])[i])
    assert np.mean(drifting) > np.mean(steady)


def test_agent_normalizer_falls_back_to_pooled():
    x = np.array([[1.0, 2.0], [3.0, 2.0], [10.0, 0.0], [12.0, 4.0]])
    norm = AgentNormalizer().fit(x, ["a", "a", "b", "b"])
    out = norm.transform(x, ["a", "a", "b", "b"])
    np.testing.assert_allclose(out[:2, 0], [-1, 1])
    assert norm.transform(x[:1], ["unseen"]).shape == (1, 2)


def test_linear_models_recover_signal():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 3))
    p = 1 / (1 + np.exp(-(2 * x[:, 0] - x[:, 1])))
    y = (rng.random(500) < p).astype(float)
    clf = LogisticRegression(l2=0.1).fit(x, y)
    assert np.corrcoef(clf.predict_proba(x), p)[0, 1] > 0.95
    soft = LogisticRegression(l2=0.1).fit(x, p)
    assert np.abs(soft.predict_proba(x) - p).mean() < 0.02
    reg = RidgeRegression(l2=0.01).fit(x, 3 * x[:, 2] + 1)
    assert np.abs(reg.predict(x) - (3 * x[:, 2] + 1)).max() < 0.01
