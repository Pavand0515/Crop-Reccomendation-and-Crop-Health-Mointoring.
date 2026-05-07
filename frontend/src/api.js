const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, options);
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json") ? await response.json() : await response.text();

  if (!response.ok) {
    const message = typeof payload === "object" && payload.detail ? payload.detail : "Request failed";
    throw new Error(message);
  }
  return payload;
}

export function getHealth() {
  return request("/api/health");
}

export function getOptions() {
  return request("/api/options");
}

export function getOverview() {
  return request("/api/overview");
}

export function getInsights() {
  return request("/api/insights");
}

export function postJson(path, data) {
  return request(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}

export function scanDisease(file) {
  const formData = new FormData();
  formData.append("file", file);
  return request("/api/disease-scan", {
    method: "POST",
    body: formData,
  });
}

export function chartUrl(fileName) {
  return `${API_BASE}/api/charts/${fileName}`;
}
