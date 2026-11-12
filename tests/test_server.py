import gzip
import json
import threading
import urllib.error
import urllib.request

import pytest

from horizonguard.monitor import server as srv


@pytest.fixture(scope="module")
def base_url(small_bench, tmp_path_factory):
    static = tmp_path_factory.mktemp("static")
    (static / "index.html").write_text("<!doctype html><title>HG</title>", encoding="utf-8")
    (static / "assets").mkdir()
    (static / "assets" / "app-abc123.js").write_text("console.log(1)", encoding="utf-8")
    state = srv.AppState(small_bench, static, ab_seeds=3)
    httpd = srv.ThreadingHTTPServer(("127.0.0.1", 0), srv.make_handler(state))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def get(url, headers=None):
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = resp.read()
        if resp.headers.get("Content-Encoding") == "gzip":
            body = gzip.decompress(body)
        return resp.status, resp.headers, body


def test_health_and_options(base_url):
    status, _, body = get(f"{base_url}/api/health")
    assert status == 200 and json.loads(body)["status"] == "ok"
    opts = json.loads(get(f"{base_url}/api/options")[2])
    assert "support-desk" in opts["envs"] and "fjord-react" in opts["agents"]


def test_runs_are_paginated_and_filterable(base_url):
    page = json.loads(get(f"{base_url}/api/runs?page=1&size=4")[2])
    assert page["total"] == 9 and len(page["items"]) == 4 and page["pages"] == 3
    hg = json.loads(get(f"{base_url}/api/runs?policy=horizonguard&size=50")[2])
    assert all(i["policy"] == "horizonguard" for i in hg["items"])
    run_id = page["items"][0]["id"]
    run = json.loads(get(f"{base_url}/api/runs/{run_id}")[2])
    assert run["timeline"] and "risk" in run["timeline"][0]


def test_gzip_and_etag_revalidation(base_url):
    _, headers, _ = get(f"{base_url}/api/overview", {"Accept-Encoding": "gzip"})
    assert headers.get("Content-Encoding") == "gzip"
    tag = headers["ETag"]
    req = urllib.request.Request(f"{base_url}/api/overview", headers={"If-None-Match": tag})
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req, timeout=60)
    assert exc.value.code == 304


def test_simulate_and_validation(base_url):
    body = json.dumps(
        {
            "env": "research",
            "agent": "birch-react",
            "policy": "horizonguard",
            "seed": 2,
            "max_steps": 20,
        }
    ).encode()
    req = urllib.request.Request(
        f"{base_url}/api/simulate",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        ep = json.loads(resp.read())
    assert ep["policy"] == "horizonguard" and len(ep["timeline"]) <= 20
    bad = urllib.request.Request(f"{base_url}/api/simulate", data=b'{"env":"mars"}', method="POST")
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(bad, timeout=60)
    assert exc.value.code == 400


def test_static_caching_and_spa_fallback(base_url):
    _, headers, _ = get(f"{base_url}/assets/app-abc123.js")
    assert "immutable" in headers["Cache-Control"]
    _, headers, body = get(f"{base_url}/runs/some-deep-link")
    assert b"<title>HG</title>" in body and headers["Cache-Control"] == "no-cache"
