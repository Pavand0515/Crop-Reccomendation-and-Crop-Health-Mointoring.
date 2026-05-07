import { useState } from "react";
import { postJson } from "../api";
import DataTable from "../components/DataTable.jsx";
import FormField from "../components/FormField.jsx";

export default function CropAdvisor({ options, health }) {
  const [form, setForm] = useState({
    nitrogen: 90,
    phosphorus: 42,
    potassium: 43,
    temperature: 20.88,
    humidity: 82,
    ph: 6.5,
    rainfall: 202.94,
    state: options.states[0] || "",
    month: 6,
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
      setResult(await postJson("/api/crop-advisor", form));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="page-stack">
      {!health.cropAdvisorAvailable ? <div className="notice danger">Crop model artifacts are missing.</div> : null}
      <form className="form-grid panel" onSubmit={submit}>
        <FormField label="Nitrogen"><input type="number" min="0" value={form.nitrogen} onChange={(e) => update("nitrogen", Number(e.target.value))} /></FormField>
        <FormField label="Phosphorus"><input type="number" min="0" value={form.phosphorus} onChange={(e) => update("phosphorus", Number(e.target.value))} /></FormField>
        <FormField label="Potassium"><input type="number" min="0" value={form.potassium} onChange={(e) => update("potassium", Number(e.target.value))} /></FormField>
        <FormField label="Temperature"><input type="number" value={form.temperature} onChange={(e) => update("temperature", Number(e.target.value))} /></FormField>
        <FormField label="Humidity"><input type="number" min="0" max="100" value={form.humidity} onChange={(e) => update("humidity", Number(e.target.value))} /></FormField>
        <FormField label="Soil pH"><input type="number" min="0" max="14" step="0.1" value={form.ph} onChange={(e) => update("ph", Number(e.target.value))} /></FormField>
        <FormField label="Rainfall"><input type="number" min="0" value={form.rainfall} onChange={(e) => update("rainfall", Number(e.target.value))} /></FormField>
        <FormField label="State"><select value={form.state} onChange={(e) => update("state", e.target.value)}>{options.states.map((state) => <option key={state}>{state}</option>)}</select></FormField>
        <FormField label="Planning month"><select value={form.month} onChange={(e) => update("month", Number(e.target.value))}>{options.months.map((month) => <option key={month} value={month}>{month}</option>)}</select></FormField>
        <button className="primary-action" type="submit" disabled={loading}>{loading ? "Checking..." : "Show Recommendations"}</button>
      </form>
      {error ? <div className="notice danger">{error}</div> : null}
      {result ? (
        <section className="panel">
          <h2>Recommended Crops for {result.season}</h2>
          <DataTable rows={result.recommendations} />
        </section>
      ) : null}
    </section>
  );
}
