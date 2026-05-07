import { useEffect, useState } from "react";
import { getOverview } from "../api";
import BarList from "../components/BarList.jsx";
import MetricCard from "../components/MetricCard.jsx";

export default function Overview() {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    getOverview().then(setData).catch((err) => setError(err.message));
  }, []);

  if (error) return <div className="notice danger">{error}</div>;
  if (!data) return <div className="notice">Loading overview...</div>;

  return (
    <section className="page-stack">
      <div className="metrics-grid">
        <MetricCard label="Market records" value={data.metrics.marketRecords.toLocaleString()} />
        <MetricCard label="Crops" value={data.metrics.crops} />
        <MetricCard label="States" value={data.metrics.states} />
        <MetricCard label="Latest year" value={data.metrics.latestYear} />
      </div>
      <section className="panel">
        <h2>Top Market Coverage</h2>
        <BarList rows={data.stateCounts} labelKey="State" valueKey="Records" />
      </section>
    </section>
  );
}
