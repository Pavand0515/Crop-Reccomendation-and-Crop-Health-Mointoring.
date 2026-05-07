import { useState } from "react";
import { scanDisease } from "../api";
import DataTable from "../components/DataTable.jsx";

export default function DiseaseScan({ health }) {
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function submit(event) {
    event.preventDefault();
    const file = event.currentTarget.elements.leaf.files[0];
    if (!file) return;

    setPreview(URL.createObjectURL(file));
    setLoading(true);
    setError("");
    try {
      setResult(await scanDisease(file));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="page-stack">
      {!health.diseaseAvailable ? (
        <div className="notice danger">{health.diseaseError || "Disease model artifacts are unavailable."}</div>
      ) : null}
      <form className="upload-panel panel" onSubmit={submit}>
        <input id="leaf" name="leaf" type="file" accept="image/*" />
        <button className="primary-action" type="submit" disabled={loading}>{loading ? "Analyzing..." : "Analyze Leaf"}</button>
      </form>
      {error ? <div className="notice danger">{error}</div> : null}
      {preview ? <img className="leaf-preview" src={preview} alt="Uploaded leaf" /> : null}
      {result ? (
        <section className="panel">
          <h2>{result.label}</h2>
          <DataTable rows={result.top_predictions} />
        </section>
      ) : null}
    </section>
  );
}
