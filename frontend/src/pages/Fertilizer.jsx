import { useState } from "react";
import { postJson } from "../api";
import DataTable from "../components/DataTable.jsx";
import FormField from "../components/FormField.jsx";

export default function Fertilizer({ options, health }) {
  const [form, setForm] = useState({
    temperature: 25,
    moisture: 0.7,
    rainfall: 150,
    ph: 6.5,
    nitrogen: 60,
    phosphorous: 60,
    potassium: 60,
    carbon: 0.5,
    crop: options.fertilizerCrops[0] || "",
    soil: options.soils[0] || "",
  });
  const [rows, setRows] = useState([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const update = (key, value) => setForm((current) => ({ ...current, [key]: value }));

  async function submit(event) {
    event.preventDefault();
    setLoading(true);
    setError("");
    try {
      const result = await postJson("/api/fertilizer", form);
      setRows(result.recommendations);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="page-stack">
      {!health.fertilizerAvailable ? <div className="notice danger">Fertilizer data is unavailable.</div> : null}
      <form className="form-grid panel" onSubmit={submit}>
        <FormField label="Temperature"><input type="number" value={form.temperature} onChange={(e) => update("temperature", Number(e.target.value))} /></FormField>
        <FormField label="Moisture"><input type="number" min="0" step="0.01" value={form.moisture} onChange={(e) => update("moisture", Number(e.target.value))} /></FormField>
        <FormField label="Rainfall"><input type="number" value={form.rainfall} onChange={(e) => update("rainfall", Number(e.target.value))} /></FormField>
        <FormField label="Soil pH"><input type="number" min="0" max="14" step="0.1" value={form.ph} onChange={(e) => update("ph", Number(e.target.value))} /></FormField>
        <FormField label="Nitrogen"><input type="number" min="0" value={form.nitrogen} onChange={(e) => update("nitrogen", Number(e.target.value))} /></FormField>
        <FormField label="Phosphorous"><input type="number" min="0" value={form.phosphorous} onChange={(e) => update("phosphorous", Number(e.target.value))} /></FormField>
        <FormField label="Potassium"><input type="number" min="0" value={form.potassium} onChange={(e) => update("potassium", Number(e.target.value))} /></FormField>
        <FormField label="Carbon"><input type="number" min="0" step="0.1" value={form.carbon} onChange={(e) => update("carbon", Number(e.target.value))} /></FormField>
        <FormField label="Crop"><select value={form.crop} onChange={(e) => update("crop", e.target.value)}>{options.fertilizerCrops.map((crop) => <option key={crop}>{crop}</option>)}</select></FormField>
        <FormField label="Soil"><select value={form.soil} onChange={(e) => update("soil", e.target.value)}>{options.soils.map((soil) => <option key={soil}>{soil}</option>)}</select></FormField>
        <button className="primary-action" type="submit" disabled={loading}>{loading ? "Matching..." : "Get Fertilizer"}</button>
      </form>
      {error ? <div className="notice danger">{error}</div> : null}
      {rows.length ? <section className="panel"><h2>Top Fertilizer Matches</h2><DataTable rows={rows} /></section> : null}
    </section>
  );
}
