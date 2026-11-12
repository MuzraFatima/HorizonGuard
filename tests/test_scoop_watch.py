from datetime import date

from horizonguard.literature import scoop_watch as sw

FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2611.00001v2</id>
    <published>2026-11-10T00:00:00Z</published>
    <title>Forecasting the Counterfactual Effect of Goal Reminders
      on Drifting Agents</title>
    <summary>We forecast, per prefix, the intervention effect of reminders on goal drift.</summary>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2611.00002v1</id>
    <published>2026-11-09T00:00:00Z</published>
    <title>A Survey of Tool Use</title>
    <summary>We survey tool use in language models.</summary>
  </entry>
</feed>
"""


def test_parse_feed_strips_versions_and_whitespace():
    papers = sw.parse_feed(FEED, "q")
    assert [p.arxiv_id for p in papers] == ["2611.00001", "2611.00002"]
    assert papers[0].title.startswith("Forecasting the Counterfactual Effect of Goal Reminders on")
    assert papers[0].published == "2026-11-10"


def test_trigger_a_fires_only_on_matching_paper():
    first, second = sw.parse_feed(FEED, "q")
    assert any(t.startswith("a:") for t in sw.triggers_for(first))
    assert sw.triggers_for(second) == []


def test_trigger_b_needs_all_groups():
    harm = sw.Paper("x", "Certified bounds on harm from intervening", "conformal", "", "q")
    assert any(t.startswith("b:") for t in sw.triggers_for(harm))
    partial = sw.Paper("y", "Harm from intervening", "an empirical study only", "", "q")
    assert not any(t.startswith("b:") for t in sw.triggers_for(partial))


def test_build_url_targets_arxiv():
    url = sw.build_url("goal drift forecasting")
    assert url.startswith(sw.ARXIV_API)
    assert "all%3Agoal+AND+all%3Adrift" in url


def test_run_deduplicates_and_skips_logged(tmp_path):
    log = tmp_path / "log.md"
    log.write_text("# log\n<!-- entries -->\n- arXiv 2611.00002: old\n", encoding="utf-8")

    def fake_fetch(query):
        return sw.parse_feed(FEED, query)

    papers, entry = sw.run(log, today=date(2026, 11, 12), fetcher=fake_fetch)
    assert [p.arxiv_id for p in papers] == ["2611.00001"]
    assert "Triggers: 1" in entry
    sw.append_entry(log, entry)
    text = log.read_text(encoding="utf-8")
    assert text.index("2026-11-12") < text.index("2611.00002: old")


def test_run_survives_network_errors(tmp_path):
    def broken(query):
        raise OSError("offline")

    papers, entry = sw.run(tmp_path / "log.md", today=date(2026, 11, 12), fetcher=broken)
    assert papers == []
    assert "fetch problem" in entry
