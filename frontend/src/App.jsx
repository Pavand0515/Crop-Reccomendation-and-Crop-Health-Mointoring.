import { useEffect, useMemo, useState } from "react";
import { BarChart3, Leaf, LineChart, Microscope, Sprout, TestTube2 } from "lucide-react";
import { getHealth, getOptions } from "./api";
import Overview from "./pages/Overview.jsx";
import CropAdvisor from "./pages/CropAdvisor.jsx";
import Fertilizer from "./pages/Fertilizer.jsx";
import PriceForecast from "./pages/PriceForecast.jsx";
import DiseaseScan from "./pages/DiseaseScan.jsx";
import Insights from "./pages/Insights.jsx";

const pages = [
  { id: "overview", label: "Overview", icon: BarChart3 },
  { id: "crop", label: "Crop Advisor", icon: Sprout },
  { id: "fertilizer", label: "Fertilizer", icon: TestTube2 },
  { id: "price", label: "Price Forecast", icon: LineChart },
  { id: "disease", label: "Disease Scan", icon: Microscope },
  { id: "insights", label: "Insights", icon: Leaf },
];

export default function App() {
  const [activePage, setActivePage] = useState("overview");
  const [options, setOptions] = useState(null);
  const [health, setHealth] = useState(null);
  const [bootError, setBootError] = useState("");

  useEffect(() => {
    Promise.all([getOptions(), getHealth()])
      .then(([optionData, healthData]) => {
        setOptions(optionData);
        setHealth(healthData);
      })
      .catch((error) => setBootError(error.message));
  }, []);

  const statusText = useMemo(() => {
    if (!health) return "Connecting";
    const ready = [
      health.cropAdvisorAvailable,
      health.priceForecastAvailable,
      health.fertilizerAvailable,
      health.diseaseAvailable,
    ].filter(Boolean).length;
    return `${ready}/4 models ready`;
  }, [health]);

  const renderPage = () => {
    if (bootError) return <div className="notice danger">{bootError}</div>;
    if (!options || !health) return <div className="notice">Loading farm intelligence...</div>;
    if (activePage === "overview") return <Overview />;
    if (activePage === "crop") return <CropAdvisor options={options} health={health} />;
    if (activePage === "fertilizer") return <Fertilizer options={options} health={health} />;
    if (activePage === "price") return <PriceForecast options={options} health={health} />;
    if (activePage === "disease") return <DiseaseScan health={health} />;
    return <Insights />;
  };

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">AI</div>
          <div>
            <strong>AgriIntel</strong>
            <span>Crop planning suite</span>
          </div>
        </div>
        <nav className="nav-list" aria-label="Dashboard sections">
          {pages.map((page) => {
            const Icon = page.icon;
            return (
              <button
                className={activePage === page.id ? "nav-item active" : "nav-item"}
                key={page.id}
                onClick={() => setActivePage(page.id)}
                type="button"
              >
                <Icon size={18} />
                <span>{page.label}</span>
              </button>
            );
          })}
        </nav>
      </aside>
      <main className="content">
        <header className="topbar">
          <div>
            <p className="kicker">Agriculture decision support</p>
            <h1>{pages.find((page) => page.id === activePage)?.label}</h1>
          </div>
          <div className="status-pill">{statusText}</div>
        </header>
        {renderPage()}
      </main>
    </div>
  );
}
