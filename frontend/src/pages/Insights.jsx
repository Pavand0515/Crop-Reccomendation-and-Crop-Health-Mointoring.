import { useEffect, useState } from "react";
import { chartUrl, getInsights } from "../api";
import DataTable from "../components/DataTable.jsx";

const tableLabels = {
  crop_profitability: "Crop Profitability",
  state_profitability: "State Profitability",
  risk_reward_matrix: "Risk Reward Matrix",
  market_stability: "Market Stability",
  seasonal_analysis: "Seasonal Analysis",
};

export default function Insights() {
  const [data, setData] = useState(null);
  const [activeTable, setActiveTable] = useState("crop_profitability");
  const [error, setError] = useState("");

  useEffect(() => {
    getInsights().then(setData).catch((err) => setError(err.message));
  }, []);

  if (error) return <div className="notice danger">{error}</div>;
  if (!data) return <div className="notice">Loading insights...</div>;

  return (
    <section className="page-stack">
      <section className="panel">
        <div className="tabs">
          {Object.keys(tableLabels).map((key) => (
            <button
              className={activeTable === key ? "active" : ""}
              key={key}
              onClick={() => setActiveTable(key)}
              type="button"
            >
              {tableLabels[key]}
            </button>
          ))}
        </div>
        <DataTable rows={data.tables[activeTable]} />
      </section>
      <section className="chart-grid">
        {data.charts.map((chart) => (
          <figure className="chart-card" key={chart}>
            <img src={chartUrl(chart)} alt={chart.replaceAll("_", " ").replace(".png", "")} />
            <figcaption>{chart.replaceAll("_", " ").replace(".png", "")}</figcaption>
          </figure>
        ))}
      </section>
    </section>
  );
}
