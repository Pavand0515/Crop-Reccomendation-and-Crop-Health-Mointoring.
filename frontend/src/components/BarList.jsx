export default function BarList({ rows, labelKey, valueKey }) {
  const maxValue = Math.max(...rows.map((row) => Number(row[valueKey]) || 0), 1);

  return (
    <div className="bar-list">
      {rows.map((row) => (
        <div className="bar-row" key={row[labelKey]}>
          <span>{row[labelKey]}</span>
          <div className="bar-track">
            <div style={{ width: `${((Number(row[valueKey]) || 0) / maxValue) * 100}%` }} />
          </div>
          <strong>{row[valueKey]}</strong>
        </div>
      ))}
    </div>
  );
}
