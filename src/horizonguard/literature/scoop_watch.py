"""Weekly scoop-watch against arXiv.

The blueprint fixes six weekly queries and two triggers. A paper triggers the
two-week response protocol when it (a) forecasts per-prefix intervention effect
for goal drift, or (b) certifies intervention-induced harm. Trigger detection
here is a keyword screen that flags candidates for a human read; it never
decides a pivot by itself.
"""

from __future__ import annotations

import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date
from pathlib import Path

ARXIV_API = "https://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"

QUERIES: tuple[str, ...] = (
    "goal drift forecasting",
    "agent failure prediction",
    "intervention effect agents",
    "counterfactual fork rollouts agent",
    "conformal e-process agent monitoring",
    "agent monitor transfer",
)

# Each trigger needs at least one term from every group to fire.
TRIGGERS: dict[str, tuple[tuple[str, ...], ...]] = {
    "a: per-prefix intervention effect for drift": (
        ("goal drift", "drift", "goal misalignment", "goal deviation"),
        ("intervention effect", "treatment effect", "counterfactual", "uplift", "rescue"),
        ("prefix", "per-step", "step-level", "early", "forecast"),
    ),
    "b: certified intervention-induced harm": (
        ("intervention", "intervening", "steering", "correction"),
        ("harm", "disruption", "degrad", "hurt"),
        ("certif", "guarantee", "conformal", "risk control", "finite-sample"),
    ),
}


@dataclass(frozen=True)
class Paper:
    arxiv_id: str
    title: str
    summary: str
    published: str
    query: str

    @property
    def text(self) -> str:
        return f"{self.title} {self.summary}".lower()


def build_url(query: str, max_results: int = 25) -> str:
    terms = " AND ".join(f"all:{word}" for word in query.split())
    params = {
        "search_query": terms,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "max_results": str(max_results),
    }
    return f"{ARXIV_API}?{urllib.parse.urlencode(params)}"


def parse_feed(xml_text: str, query: str) -> list[Paper]:
    root = ET.fromstring(xml_text)
    papers = []
    for entry in root.findall(f"{ATOM}entry"):
        raw_id = (entry.findtext(f"{ATOM}id") or "").rsplit("/", 1)[-1]
        arxiv_id = re.sub(r"v\d+$", "", raw_id)
        papers.append(
            Paper(
                arxiv_id=arxiv_id,
                title=" ".join((entry.findtext(f"{ATOM}title") or "").split()),
                summary=" ".join((entry.findtext(f"{ATOM}summary") or "").split()),
                published=(entry.findtext(f"{ATOM}published") or "")[:10],
                query=query,
            )
        )
    return papers


def triggers_for(paper: Paper) -> list[str]:
    text = paper.text
    return [
        name
        for name, groups in TRIGGERS.items()
        if all(any(term in text for term in group) for group in groups)
    ]


def fetch(query: str, timeout: float = 20.0) -> list[Paper]:
    request = urllib.request.Request(build_url(query), headers={"User-Agent": "horizonguard"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return parse_feed(response.read().decode("utf-8"), query)


def known_ids(log_path: Path) -> set[str]:
    if not log_path.exists():
        return set()
    return set(re.findall(r"arXiv (\d{4}\.\d{4,5})", log_path.read_text(encoding="utf-8")))


def render_entry(day: date, papers: list[Paper], errors: list[str]) -> str:
    lines = [f"## {day.isoformat()}", ""]
    flagged = [(p, triggers_for(p)) for p in papers]
    hits = [(p, t) for p, t in flagged if t]
    lines.append(f"Queries run: {len(QUERIES)}. New papers: {len(papers)}. Triggers: {len(hits)}.")
    lines.append("")
    if hits:
        lines.append("**Triggered (start the two-week response protocol):**")
        lines.append("")
        for paper, names in hits:
            lines.append(f"- arXiv {paper.arxiv_id}: {paper.title} ({'; '.join(names)})")
        lines.append("")
    rest = [p for p, t in flagged if not t]
    if rest:
        lines.append("Other new papers:")
        lines.append("")
        lines += [f"- arXiv {p.arxiv_id}: {p.title} [{p.query}]" for p in rest]
        lines.append("")
    for err in errors:
        lines.append(f"> fetch problem: {err}")
    return "\n".join(lines).rstrip() + "\n\n"


def run(log_path: Path, today: date | None = None, fetcher=fetch) -> tuple[list[Paper], str]:
    """Run all queries, keep papers not already logged, and return the new entry."""
    seen = known_ids(log_path)
    fresh: dict[str, Paper] = {}
    errors: list[str] = []
    for query in QUERIES:
        try:
            for paper in fetcher(query):
                if paper.arxiv_id not in seen:
                    fresh.setdefault(paper.arxiv_id, paper)
        except Exception as exc:  # network trouble should not kill the weekly log
            errors.append(f"{query}: {exc}")
    papers = sorted(fresh.values(), key=lambda p: p.published, reverse=True)
    return papers, render_entry(today or date.today(), papers, errors)


def append_entry(log_path: Path, entry: str) -> None:
    text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    marker = "<!-- entries -->\n"
    if marker in text:
        head, tail = text.split(marker, 1)
        text = head + marker + "\n" + entry + tail.lstrip("\n")
    else:
        text = text + entry
    log_path.write_text(text, encoding="utf-8", newline="\n")
