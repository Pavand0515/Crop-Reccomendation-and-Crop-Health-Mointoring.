import { useState } from "react";
import { postJson } from "../api";
import FormField from "../components/FormField.jsx";
import MetricCard from "../components/MetricCard.jsx";

export default function PriceForecast({ options, health }) {
  const [form, setForm] = useState({
    state: options.states[0] || "",
    crop: options.crops[0] || "",
    season: "Kharif",
  });
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const update = (key, value) => setForm((current) => ({ ...current, [key]: value }));

  async function submit(event) {
    event.preventDefault();
    setLoading(true);
    setError("");
    try {
      setResult(await postJson("/api/price-forecast", form));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  const historyMax = Math.max(...(result?.history || []).map((point) => point.price), 1);

  return (
    <section className="page-stack">
      {!health.priceForecastAvailable ? <div className="notice danger">Price forecast model artifacts are missing.</div> : null}
      <form className="inline-form panel" onSubmit={submit}>
        <FormField label="State"><select value={form.state} onChange={(e) => update("state", e.target.value)}>{options.states.map((state) => <option key={state}>{state}</option>)}</select></FormField>
        <FormField label="Season"><select value={form.season} onChange={(e) => update("season", e.target.value)}>{options.seasons.map((season) => <option key={season}>{season}</option>)}</select></FormField>
        <FormField label="Crop"><select value={form.crop} onChange={(e) => update("crop", e.target.value)}>{options.crops.map((crop) => <option key={crop}>{crop}</option>)}</select></FormField>
        <button className="primary-action" type="submit" disabled={loading}>{loading ? "Forecasting..." : "Forecast Price"}</button>
      </form>
      {error ? <div className="notice danger">{error}</div> : null}
      {result ? (
        <>
          <div className="metrics-grid">
            <MetricCard label="Expected price" value={`Rs. ${result.expectedPrice.toLocaleString()}`} />
            <MetricCard label="Trend" value={result.trend} />
            <MetricCard label="Profit margin" value={`${result.profitMargin}%`} />
          </div>
          <section className="panel">
            <h2>Recent Price History</h2>
            <div className="spark-bars">
              {result.history.map((point) => (
                <div className="spark-bar" key={point.date} title={`${point.date}: Rs. ${point.price}`}>
                  <span style={{ height: `${Math.max((point.price / historyMax) * 100, 8)}%` }} />
                </div>
              ))}
            </div>
          </section>
        </>
      ) : null}
    </section>
  );
}
