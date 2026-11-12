import { Link, useParams } from "react-router-dom";
import type { Episode } from "../api";
import { EpisodeView } from "../components/EpisodeView";
import { CardSkeleton, ErrorNote, PageHead } from "../components/ui";
import { useApi } from "../hooks";
import { POLICY_LABEL } from "../labels";

export default function RunDetail() {
  const { id = "" } = useParams();
  const { data, error, loading } = useApi<Episode>(`/api/runs/${encodeURIComponent(id)}`);
  const sibling = (policy: string) => id.replace(/-(none|probability|horizonguard)$/, `-${policy}`);

  return (
    <div className="page">
      <PageHead
        eyebrow={data ? POLICY_LABEL[data.policy] : "Run"}
        title={id}
        aside={
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            {(["none", "probability", "horizonguard"] as const).map((p) => (
              <Link
                key={p}
                to={`/runs/${sibling(p)}`}
                className={`btn ${data?.policy === p ? "" : "btn-ghost"}`}
                style={{ textDecoration: "none" }}
              >
                {POLICY_LABEL[p]}
              </Link>
            ))}
          </div>
        }
      />
      {error && <ErrorNote message={error} />}
      {loading || !data ? <CardSkeleton chart lines={2} /> : <EpisodeView episode={data} />}
      <p style={{ marginTop: 18 }}>
        <Link to="/runs" className="card-sub">
          ← back to all runs
        </Link>
      </p>
    </div>
  );
}
