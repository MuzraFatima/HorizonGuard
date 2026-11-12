"""HTTP API and static host for the HorizonGuard dashboard (standard library only).

Performance features:

* ETag on every response, so ``If-None-Match`` gets a 304 and the client reuses
  its copy;
* ``Cache-Control``: hashed static assets are immutable for a year, API reads are
  cacheable for a short window, and ``index.html`` is always revalidated;
* gzip for any payload over 1 KB when the client accepts it, and floats rounded
  to 4 places before encoding;
* server-side caching: the A/B library is computed once and stored on disk,
  simulations are memoised in an LRU keyed by their parameters;
* pagination and filtering on the runs list, so large libraries stay cheap.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import mimetypes
import threading
from collections import OrderedDict
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from horizonguard import __version__
from horizonguard.agents import AGENT_NAMES
from horizonguard.bench.grid import DRIFT_TYPES
from horizonguard.envs import ENVIRONMENTS
from horizonguard.monitor.closed_loop import (
    SupportDeskEnv,
    ab_test,
    demo_agent,
    demo_env,
    fresh_agent,
    fresh_env,
    run_closed_loop,
)
from horizonguard.monitor.live import LiveMonitor, load_or_build

POLICIES = ("none", "probability", "horizonguard")
DEMO_AGENTS = (*[a for a in AGENT_NAMES if a != "scripted"], "fjord-react", "fjord-plan")
DEMO_ENVS = (*ENVIRONMENTS, SupportDeskEnv.name)
LIBRARY_FILE = "dashboard-library.json"


def _round(obj: Any) -> Any:
    if isinstance(obj, float):
        return round(obj, 4)
    if isinstance(obj, dict):
        return {k: _round(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round(v) for v in obj]
    return obj


def _summary(run_id: str, ep: dict[str, Any]) -> dict[str, Any]:
    risks = [e.get("risk", 0.0) for e in ep["timeline"]]
    return {
        "id": run_id,
        "env": ep["env"],
        "agent": ep["agent"],
        "policy": ep["policy"],
        "seed": ep["seed"],
        "violated": ep["violated"],
        "succeeded": ep["succeeded"],
        "interventions": ep["interventions"],
        "length": len(ep["timeline"]),
        "peak_risk": max(risks) if risks else 0.0,
    }


class AppState:
    def __init__(self, bench_dir: str | Path, static_dir: str | Path | None, ab_seeds: int = 40):
        self.bench_dir = Path(bench_dir)
        self.static_dir = Path(static_dir) if static_dir else None
        self.ab_seeds = ab_seeds
        self.artifacts = load_or_build(self.bench_dir)
        self._lock = threading.Lock()
        self._library: dict[str, Any] | None = None
        self._sim_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def monitor(self) -> LiveMonitor:
        return LiveMonitor(self.artifacts)

    # ----- the cached library of closed-loop runs --------------------------------
    def library(self) -> dict[str, Any]:
        with self._lock:
            if self._library is not None:
                return self._library
            path = self.bench_dir / LIBRARY_FILE
            if path.exists():
                self._library = json.loads(path.read_text(encoding="utf-8"))
                return self._library
            monitor = self.monitor()
            episodes: dict[str, Any] = {}
            for seed in range(self.ab_seeds):
                for policy in POLICIES:
                    ep = run_closed_loop(fresh_env(seed), fresh_agent(seed), seed, policy, monitor)
                    episodes[f"ab-{seed:03d}-{policy}"] = _round(ep)
            ab = ab_test(monitor, range(self.ab_seeds))
            self._library = {
                "episodes": episodes,
                "ab": _round(ab.summary),
                "ab_rows": _round(ab.rows),
            }
            path.write_text(json.dumps(self._library), encoding="utf-8")
            return self._library

    def simulate(self, params: dict[str, Any]) -> dict[str, Any]:
        env = str(params.get("env", SupportDeskEnv.name))
        agent = str(params.get("agent", "fjord-react"))
        drift = str(params.get("drift_type", "nudge"))
        policy = str(params.get("policy", "horizonguard"))
        seed = int(params.get("seed", 0))
        steps = min(max(int(params.get("max_steps", 50)), 10), 80)
        if env not in DEMO_ENVS or agent not in DEMO_AGENTS or drift not in DRIFT_TYPES:
            raise ValueError("unknown environment, agent or drift type")
        if policy not in POLICIES:
            raise ValueError("unknown policy")
        key = json.dumps([env, agent, drift, policy, seed, steps])
        with self._lock:
            if key in self._sim_cache:
                self._sim_cache.move_to_end(key)
                return self._sim_cache[key]
        ep = run_closed_loop(
            demo_env(env, seed, drift, steps), demo_agent(agent), seed, policy, self.monitor()
        )
        result = _round(ep)
        with self._lock:
            self._sim_cache[key] = result
            while len(self._sim_cache) > 128:
                self._sim_cache.popitem(last=False)
        return result

    # ----- read endpoints -------------------------------------------------------------
    def runs(self, query: dict[str, list[str]]) -> dict[str, Any]:
        lib = self.library()
        items = [_summary(rid, ep) for rid, ep in sorted(lib["episodes"].items())]
        q = (query.get("q") or [""])[0].strip().lower()
        policy = (query.get("policy") or [""])[0]
        outcome = (query.get("outcome") or [""])[0]
        if q:
            items = [i for i in items if q in i["id"] or q in i["agent"] or q in i["env"]]
        if policy:
            items = [i for i in items if i["policy"] == policy]
        if outcome == "violated":
            items = [i for i in items if i["violated"]]
        elif outcome == "clean":
            items = [i for i in items if not i["violated"]]
        page = max(int((query.get("page") or ["1"])[0]), 1)
        size = min(max(int((query.get("size") or ["12"])[0]), 1), 100)
        start = (page - 1) * size
        return {
            "items": items[start : start + size],
            "page": page,
            "size": size,
            "total": len(items),
            "pages": max(1, -(-len(items) // size)),
        }

    def run(self, run_id: str) -> dict[str, Any] | None:
        return self.library()["episodes"].get(run_id)

    def overview(self) -> dict[str, Any]:
        lib = self.library()
        return {
            "version": __version__,
            "certificate": _round(self.artifacts.certificate),
            "ab": lib["ab"],
            "trained_on": self.artifacts.trained_on,
            "findings": FINDINGS,
        }

    def options(self) -> dict[str, Any]:
        return {
            "envs": list(DEMO_ENVS),
            "agents": list(DEMO_AGENTS),
            "drift_types": list(DRIFT_TYPES),
            "policies": list(POLICIES),
        }

    def guarantees(self) -> dict[str, Any]:
        return {
            "certificate": _round(self.artifacts.certificate),
            "studies": GUARANTEE_STUDIES,
            "eprocess_m0": round(self.artifacts.eprocess_m0, 4),
        }


FINDINGS = [
    {
        "rq": "RQ1",
        "claim": "Multi-state hazard beats single-event forecasts on lead time",
        "verdict": "not supported",
        "evidence": "-1.5 steps vs single-event (CI excludes 0)",
    },
    {
        "rq": "RQ2",
        "claim": "Effect-aware policy beats probability-only alarms",
        "verdict": "supported",
        "evidence": "net benefit 0.076 vs 0.001 on fresh forks",
    },
    {
        "rq": "RQ2",
        "claim": "Wins in 50% of cells where probability alarms fail",
        "verdict": "not met",
        "evidence": "34% of cells; no worse in 32 of 35",
    },
    {
        "rq": "RQ3",
        "claim": "Disruption bound is valid and non-vacuous",
        "verdict": "supported",
        "evidence": "0% violations in 200 resamples; 96% rescues kept",
    },
    {
        "rq": "RQ4",
        "claim": "Drift signatures transfer to unseen agents",
        "verdict": "supported",
        "evidence": "median AUROC drop 0.03",
    },
    {
        "rq": "RQ4",
        "claim": "Intervention response does not transfer",
        "verdict": "not supported",
        "evidence": "effects transferred better than risk",
    },
    {
        "rq": "RQ5",
        "claim": "Lead time follows a simple scaling law",
        "verdict": "not supported",
        "evidence": "inverted U in drift strength",
    },
    {
        "rq": "RQ5",
        "claim": "Randomised thresholds blunt stealth drift",
        "verdict": "supported",
        "evidence": "evasion 100% to 44%",
    },
]

GUARANTEE_STUDIES = {
    "sample_complexity": [
        {"n": 20, "vacuous": 1.0, "violations": 0.0, "retention": 0.0},
        {"n": 40, "vacuous": 1.0, "violations": 0.0, "retention": 0.0},
        {"n": 80, "vacuous": 0.0, "violations": 0.0, "retention": 0.15},
        {"n": 120, "vacuous": 0.0, "violations": 0.0, "retention": 0.92},
    ],
    "coverage": {"resamples": 200, "ltt_violation_rate": 0.0, "crc_mean_risk": 0.0102},
    "time_uniform": {
        "eprocess_fpr": 0.0,
        "eprocess_detection": 0.58,
        "eprocess_lead": 6.35,
        "fixed_detection": 0.5,
        "fixed_lead": 6.12,
    },
}


@lru_cache(maxsize=256)
def _etag(body: bytes) -> str:
    return '"' + hashlib.sha1(body).hexdigest()[:20] + '"'


def make_handler(state: AppState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = f"HorizonGuard/{__version__}"

        def log_message(self, fmt: str, *args: Any) -> None:  # quiet by default
            return

        # ----- response helpers ------------------------------------------------------
        def _send(self, status: int, body: bytes, ctype: str, cache: str) -> None:
            tag = _etag(body)
            if self.headers.get("If-None-Match") == tag and status == 200:
                self.send_response(HTTPStatus.NOT_MODIFIED)
                self.send_header("ETag", tag)
                self.send_header("Cache-Control", cache)
                self.end_headers()
                return
            gz = "gzip" in (self.headers.get("Accept-Encoding") or "") and len(body) > 1024
            payload = gzip.compress(body, compresslevel=6) if gz else body
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("ETag", tag)
            self.send_header("Cache-Control", cache)
            self.send_header("Vary", "Accept-Encoding")
            self.send_header("X-Content-Type-Options", "nosniff")
            if gz:
                self.send_header("Content-Encoding", "gzip")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(payload)

        def _json(self, data: Any, status: int = 200, cache: str = "public, max-age=30") -> None:
            body = json.dumps(data, separators=(",", ":")).encode("utf-8")
            self._send(status, body, "application/json; charset=utf-8", cache)

        def _error(self, status: int, message: str) -> None:
            self._json({"error": message}, status, cache="no-store")

        # ----- routing -----------------------------------------------------------------
        def do_GET(self) -> None:
            url = urlparse(self.path)
            path, query = url.path, parse_qs(url.query)
            try:
                if path == "/api/health":
                    return self._json({"status": "ok", "version": __version__}, cache="no-store")
                if path == "/api/overview":
                    return self._json(state.overview())
                if path == "/api/options":
                    return self._json(state.options(), cache="public, max-age=3600")
                if path == "/api/runs":
                    return self._json(state.runs(query))
                if path.startswith("/api/runs/"):
                    run = state.run(path.rsplit("/", 1)[-1])
                    return self._json(run) if run else self._error(404, "run not found")
                if path == "/api/ab":
                    lib = state.library()
                    return self._json({"summary": lib["ab"], "rows": lib["ab_rows"]})
                if path == "/api/guarantees":
                    return self._json(state.guarantees())
                if path.startswith("/api/"):
                    return self._error(404, "unknown endpoint")
                return self._static(path)
            except ValueError as exc:
                return self._error(400, str(exc))

        do_HEAD = do_GET

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/api/simulate":
                return self._error(404, "unknown endpoint")
            try:
                length = min(int(self.headers.get("Content-Length") or 0), 10_000)
                params = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(params, dict):
                    raise ValueError("body must be a JSON object")
                return self._json(state.simulate(params), cache="private, max-age=300")
            except (ValueError, json.JSONDecodeError) as exc:
                return self._error(400, str(exc))

        def _static(self, path: str) -> None:
            root = state.static_dir
            if root is None or not root.exists():
                return self._error(404, "dashboard not built; run npm run build in dashboard/")
            target = (root / path.lstrip("/")).resolve()
            if not str(target).startswith(str(root.resolve())) or not target.is_file():
                target = root / "index.html"  # single-page app fallback
            ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            if target.name == "index.html":
                cache = "no-cache"
            elif "/assets/" in target.as_posix():
                cache = "public, max-age=31536000, immutable"  # content-hashed by Vite
            else:
                cache = "public, max-age=3600"
            self._send(200, target.read_bytes(), ctype, cache)

    return Handler


def serve(
    bench_dir: str | Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    static_dir: str | Path | None = "dashboard/dist",
    warm: bool = True,
) -> ThreadingHTTPServer:
    state = AppState(bench_dir, static_dir)
    if warm:
        threading.Thread(target=state.library, daemon=True).start()
    return ThreadingHTTPServer((host, port), make_handler(state))
