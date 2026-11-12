import { lazy, Suspense } from "react";
import { NavLink, Route, Routes } from "react-router-dom";
import { IconBook, IconList, IconOverview, IconPulse, IconShield, IconSplit } from "./components/Icons";
import { CardSkeleton } from "./components/ui";

// Each page is its own chunk, loaded on first visit.
const Overview = lazy(() => import("./pages/Overview"));
const Live = lazy(() => import("./pages/Live"));
const Runs = lazy(() => import("./pages/Runs"));
const RunDetail = lazy(() => import("./pages/RunDetail"));
const ABTest = lazy(() => import("./pages/ABTest"));
const Guarantees = lazy(() => import("./pages/Guarantees"));
const Method = lazy(() => import("./pages/Method"));

const NAV = [
  { to: "/", label: "Overview", icon: IconOverview, end: true },
  { to: "/live", label: "Live monitor", icon: IconPulse },
  { to: "/runs", label: "Runs", icon: IconList },
  { to: "/ab", label: "A/B test", icon: IconSplit },
  { to: "/guarantees", label: "Guarantees", icon: IconShield },
  { to: "/method", label: "Method", icon: IconBook },
];

function PageFallback() {
  return (
    <div className="page">
      <div className="skeleton" style={{ height: 34, width: "45%", marginBottom: 26 }} />
      <div className="grid grid-3">
        <CardSkeleton lines={2} />
        <CardSkeleton lines={2} />
        <CardSkeleton lines={2} />
      </div>
    </div>
  );
}

export default function App() {
  return (
    <div className="shell">
      <aside className="sidebar">
        <NavLink to="/" className="brand" aria-label="HorizonGuard home">
          <img src="/favicon.svg" width={36} height={36} alt="" />
          <span>
            <span className="brand-name">HorizonGuard</span>
            <span className="brand-tag">goal-drift monitor</span>
          </span>
        </NavLink>
        <nav className="nav" aria-label="Main">
          <div className="nav-label">Monitor</div>
          {NAV.map(({ to, label, icon: Icon, end }) => (
            <NavLink key={to} to={to} end={end}>
              <Icon />
              {label}
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-foot">
          <strong>Simulation study.</strong> Synthetic agents on DriftBench-CF; no real model is evaluated
          here.
        </div>
      </aside>
      <main className="main">
        <Suspense fallback={<PageFallback />}>
          <Routes>
            <Route path="/" element={<Overview />} />
            <Route path="/live" element={<Live />} />
            <Route path="/runs" element={<Runs />} />
            <Route path="/runs/:id" element={<RunDetail />} />
            <Route path="/ab" element={<ABTest />} />
            <Route path="/guarantees" element={<Guarantees />} />
            <Route path="/method" element={<Method />} />
            <Route path="*" element={<div className="page empty">Page not found.</div>} />
          </Routes>
        </Suspense>
      </main>
    </div>
  );
}
