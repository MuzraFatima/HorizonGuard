import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import type { Page, RunSummary } from "../api";
import { pct, pretty } from "../api";
import { Chip, ErrorNote, PageHead, Skeleton } from "../components/ui";
import { useApi, useDebounced } from "../hooks";
import { POLICY_LABEL } from "../labels";

const SIZE = 12;

export default function Runs() {
  const [search, setSearch] = useState("");
  const [policy, setPolicy] = useState("");
  const [outcome, setOutcome] = useState("");
  const [page, setPage] = useState(1);
  const q = useDebounced(search, 300);
  const navigate = useNavigate();

  const path = useMemo(() => {
    const params = new URLSearchParams({ page: String(page), size: String(SIZE) });
    if (q) params.set("q", q);
    if (policy) params.set("policy", policy);
    if (outcome) params.set("outcome", outcome);
    return `/api/runs?${params}`;
  }, [page, q, policy, outcome]);

  const { data, error, loading } = useApi<Page<RunSummary>>(path);
  const resetPage = () => setPage(1);

  return (
    <div className="page">
      <PageHead
        eyebrow="Run library"
        title="Every closed-loop episode"
        lede="Each seed was run three times with shared randomness: without a monitor, with a probability alarm,
        and with HorizonGuard. Open any run to replay it."
      />

      <div className="card">
        <div className="controls" style={{ marginBottom: 16 }}>
          <label className="field" style={{ flex: 1, minWidth: 220 }}>
            <span>Search</span>
            <input
              className="input"
              placeholder="run id, agent or environment"
              value={search}
              onChange={(e) => {
                setSearch(e.target.value);
                resetPage();
              }}
            />
          </label>
          <label className="field">
            <span>Policy</span>
            <select
              className="select"
              value={policy}
              onChange={(e) => {
                setPolicy(e.target.value);
                resetPage();
              }}
            >
              <option value="">All policies</option>
              {Object.entries(POLICY_LABEL).map(([k, v]) => (
                <option key={k} value={k}>
                  {v}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>Outcome</span>
            <select
              className="select"
              value={outcome}
              onChange={(e) => {
                setOutcome(e.target.value);
                resetPage();
              }}
            >
              <option value="">Any outcome</option>
              <option value="violated">Violated</option>
              <option value="clean">Stayed on goal</option>
            </select>
          </label>
        </div>

        {error && <ErrorNote message={error} />}
        <table className="table">
          <thead>
            <tr>
              <th>Run</th>
              <th>Policy</th>
              <th>Agent</th>
              <th>Steps</th>
              <th>Peak risk</th>
              <th>Interventions</th>
              <th>Outcome</th>
            </tr>
          </thead>
          <tbody>
            {loading || !data
              ? Array.from({ length: 6 }, (_, i) => (
                  <tr key={i}>
                    {Array.from({ length: 7 }, (_, j) => (
                      <td key={j}>
                        <Skeleton h={12} w={j === 0 ? "80%" : "60%"} />
                      </td>
                    ))}
                  </tr>
                ))
              : data.items.map((r) => (
                  <tr
                    key={r.id}
                    className="clickable"
                    tabIndex={0}
                    onClick={() => navigate(`/runs/${r.id}`)}
                    onKeyDown={(e) => e.key === "Enter" && navigate(`/runs/${r.id}`)}
                  >
                    <td className="mono">{r.id}</td>
                    <td>{POLICY_LABEL[r.policy]}</td>
                    <td>{pretty(r.agent)}</td>
                    <td className="mono">{r.length}</td>
                    <td className="mono">{pct(r.peak_risk)}</td>
                    <td className="mono">{r.interventions}</td>
                    <td>{r.violated ? <Chip tone="bad">violated</Chip> : <Chip tone="good">on goal</Chip>}</td>
                  </tr>
                ))}
          </tbody>
        </table>
        {data && data.items.length === 0 && <div className="empty">No runs match these filters.</div>}
        {data && (
          <div className="pager">
            <span>
              {data.total} runs · page {data.page} of {data.pages}
            </span>
            <span style={{ display: "flex", gap: 8 }}>
              <button className="btn btn-ghost" disabled={page <= 1} onClick={() => setPage((p) => p - 1)}>
                Previous
              </button>
              <button
                className="btn btn-ghost"
                disabled={page >= data.pages}
                onClick={() => setPage((p) => p + 1)}
              >
                Next
              </button>
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
