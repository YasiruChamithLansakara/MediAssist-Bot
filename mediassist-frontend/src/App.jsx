import { useMemo, useState } from "react";
import "./App.css";

const API_BASE = "http://127.0.0.1:8000";

/* ------------------------------------------------------------------ */
/* Helpers */
/* ------------------------------------------------------------------ */
function clampText(text = "", maxLen = 180) {
  if (!text) return "";
  if (text.length <= maxLen) return text;
  return text.slice(0, maxLen).trimEnd() + "…";
}

function normalizeSuggestionItem(item) {
  // Backend may return: ["bzk", "acetaminophen"]
  if (typeof item === "string") return item.trim();

  // Or backend may return: [{name:"bzk", score:87.2}, ...]
  if (item && typeof item === "object") {
    const maybeName =
      item.name ?? item.label ?? item.value ?? item.drug ?? item.generic_name;
    if (typeof maybeName === "string") return maybeName.trim();
  }

  return "";
}

function statusToUi(status) {
  // ✅ Uses backend response.status directly
  if (status === "ok") return "ok";
  if (status === "low_confidence") return "low";
  if (status === "no_match") return "none";
  return "idle";
}

function badgeToneFromScore(score) {
  const s = Number(score ?? 0);
  if (s >= 95) return "good";
  if (s >= 80) return "warn";
  return "bad";
}

/* ------------------------------------------------------------------ */
/* App */
/* ------------------------------------------------------------------ */
export default function App() {
  const [drug, setDrug] = useState("");
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState(null);
  const [error, setError] = useState("");

  const [showBrandsFull, setShowBrandsFull] = useState(false);

  const [openSections, setOpenSections] = useState({
    indications: false,
    dosage: false,
    warnings: false,
    contraindications: false,
    raw: false,
  });

  const [selectedSuggestion, setSelectedSuggestion] = useState("");

  // ✅ 1) Use backend status (no guessing)
  const status = useMemo(
    () => statusToUi(response?.status),
    [response?.status],
  );

  // ✅ 2) Use best_match (not matches[0])
  const match = response?.best_match || null;

  const normalized = response?.normalized || "";

  // Prefer backend best_score; fallback to match.score
  const bestScore = Number(response?.best_score ?? match?.score ?? 0);

  // ✅ 3) Show confidence in UI (0.95 → 95%)
  const confidencePercent = useMemo(() => {
    const c = Number(response?.confidence);
    if (!Number.isFinite(c)) return null;
    return Math.round(c * 100);
  }, [response?.confidence]);

  // suggestions can still be string[] (or tolerate object[] just in case)
  const suggestions = useMemo(() => {
    const raw = response?.suggestions;
    if (!Array.isArray(raw)) return [];

    const clean = raw.map(normalizeSuggestionItem).filter(Boolean);
    return Array.from(new Set(clean));
  }, [response]);

  const confidence = useMemo(() => {
    if (!match) return { label: "", tone: "neutral" };

    const tone = badgeToneFromScore(bestScore);

    // show % if available; fallback to score
    const pctText =
      typeof confidencePercent === "number" ? `${confidencePercent}%` : null;

    const label =
      tone === "good"
        ? `High confidence match (${pctText ?? `Score: ${bestScore.toFixed(1)}`})`
        : tone === "warn"
          ? `Medium confidence — verify (${pctText ?? `Score: ${bestScore.toFixed(1)}`})`
          : `Low confidence (${pctText ?? `Score: ${bestScore.toFixed(1)}`})`;

    return { label, tone };
  }, [match, bestScore, confidencePercent]);

  const brandNames = match?.brand_names || "";
  const brandShort = clampText(brandNames, 170);

  const toggleSection = (key) => {
    setOpenSections((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const resetToggles = () => {
    setShowBrandsFull(false);
    setOpenSections({
      indications: false,
      dosage: false,
      warnings: false,
      contraindications: false,
      raw: false,
    });
  };

  const clearAll = () => {
    setDrug("");
    setResponse(null);
    setError("");
    setSelectedSuggestion("");
    resetToggles();
  };

  /* ------------------------------------------------------------------ */
  /* API calls */
  /* ------------------------------------------------------------------ */
  const lookupDrugWithValue = async (value) => {
    const q = String(value || "").trim();
    if (!q) return;

    setLoading(true);
    setError("");
    setResponse(null);
    resetToggles();

    try {
      const res = await fetch(
        `${API_BASE}/lookup?drug=${encodeURIComponent(q)}`,
      );
      if (!res.ok) {
        const txt = await res.text();
        throw new Error(`API error ${res.status}: ${txt}`);
      }
      const data = await res.json();
      setResponse(data);
    } catch (e) {
      setError(e?.message || "Failed to connect to API");
    } finally {
      setLoading(false);
    }
  };

  const lookupDrug = async () => {
    setSelectedSuggestion("");
    await lookupDrugWithValue(drug);
  };

  const applySuggestion = (s) => {
    const val = String(s || "").trim();
    if (!val) return;
    setSelectedSuggestion(val);
    setDrug(val);
    lookupDrugWithValue(val);
  };

  const onKeyDown = (e) => {
    if (e.key === "Enter") lookupDrug();
    if (e.key === "Escape") clearAll();
  };

  /* ------------------------------------------------------------------ */
  /* Render */
  /* ------------------------------------------------------------------ */
  return (
    <div className="page">
      <div className="container">
        <header className="header">
          <h1>MediAssist – Drug Lookup</h1>

          <div className="subline">
            <span className="muted">Normalized:</span>
            <span className={`chip ${normalized ? "chipOn" : "chipOff"}`}>
              {normalized || "—"}
            </span>
          </div>
        </header>

        <div className="searchRow">
          <input
            className="input"
            placeholder="Enter drug name (e.g. paracetamol)"
            value={drug}
            onChange={(e) => setDrug(e.target.value)}
            onKeyDown={onKeyDown}
          />

          <button
            className="btn btnPrimary"
            onClick={lookupDrug}
            disabled={loading || !drug.trim()}
          >
            {loading ? "Searching…" : "Search"}
          </button>

          <button
            className="btn btnGhost"
            onClick={clearAll}
            disabled={loading}
          >
            Clear
          </button>
        </div>

        {loading && (
          <div className="loadingContainer">
            <div className="spinner"></div>
            <div className="loadingText">Searching drug database...</div>
          </div>
        )}

        {response?.query && !loading && (
          <div className="apiLine">
            <span className="muted">API:</span>{" "}
            <a
              href={`${API_BASE}/lookup?drug=${encodeURIComponent(response.query)}`}
              target="_blank"
              rel="noreferrer"
            >
              {`${API_BASE}/lookup?drug=${encodeURIComponent(response.query)}`}
            </a>
          </div>
        )}

        {error && <div className="alert alertBad">❌ {error}</div>}

        {/* backend status */}
        {response && status === "none" && (
          <div className="alert alertBad">
            ❌ No confident match found. Try another spelling, brand name, or
            remove dosage.
          </div>
        )}

        {response && status === "low" && (
          <div className="alert alertWarn">
            ⚠️ Low confidence match — please select from suggestions below.
          </div>
        )}

        {response && (status === "low" || status === "none") && (
          <div className="card">
            <div className="cardTop">
              <div>
                <h2 className="title">Suggestions</h2>
                <div className="muted">
                  Pick one to search again (for <b>{response.query}</b>)
                </div>
              </div>
              {selectedSuggestion && (
                <div className="chip chipOn">
                  Selected: {selectedSuggestion}
                </div>
              )}
            </div>

            {suggestions.length ? (
              <div className="suggestGrid">
                {suggestions.map((s) => (
                  <button
                    key={s}
                    className="suggestBtn"
                    onClick={() => applySuggestion(s)}
                    type="button"
                    disabled={loading}
                    title={s}
                  >
                    <span className="suggestText">{s}</span>
                    <span className="suggestHint">Use this</span>
                  </button>
                ))}
              </div>
            ) : (
              <div className="muted" style={{ marginTop: 10 }}>
                No suggestions returned.
              </div>
            )}
          </div>
        )}

        {/* best_match + backend status */}
        {response && match && status !== "none" && (
          <div className="card">
            <div className="cardTop">
              <div>
                <h2 className="title">
                  {match.generic_name_clean || match.generic_name}
                </h2>
                <div className="muted">
                  Matched as: <b>{match.match}</b>
                </div>

                {/* ✅ tiny confidence display (95%) */}
                {typeof confidencePercent === "number" && (
                  <div className="muted" style={{ marginTop: 4 }}>
                    Confidence: <b>{confidencePercent}%</b>
                  </div>
                )}
              </div>

              <div className={`badge badge-${confidence.tone}`}>
                {confidence.tone === "good"
                  ? "✅"
                  : confidence.tone === "warn"
                    ? "⚠️"
                    : "❌"}{" "}
                {confidence.label}
              </div>
            </div>

            {bestScore < 95 && (
              <div className="alert alertWarn">
                ⚠️ Please verify this result before use.
              </div>
            )}

            <div className="grid">
              <Info label="Route" value={match.route} />
              <Info label="Drug class" value={match.drug_class} />

              <div className="infoBox span2">
                <div className="label">Brand names</div>
                <div className="value">
                  {brandNames
                    ? showBrandsFull
                      ? brandNames
                      : brandShort
                    : "—"}
                </div>
                {brandNames.length > 170 && (
                  <button
                    className="linkBtn"
                    onClick={() => setShowBrandsFull((s) => !s)}
                    type="button"
                  >
                    {showBrandsFull ? "Show less" : "Show more"}
                  </button>
                )}
              </div>

              <Info
                label="Sources"
                value={(match.sources || "—").replaceAll("|", " | ")}
              />
              <Info label="Last updated" value={match.last_updated} />
            </div>

            {match.side_effects_buckets && (
              <section className="section">
                <h3>Side Effects</h3>
                <div className="sideEffects">
                  {Object.entries(match.side_effects_buckets).map(([k, v]) => (
                    <div className="sideRow" key={k}>
                      <div className="sideKey">
                        {k.replaceAll("_", " ").toUpperCase()}
                      </div>
                      <div className="sideVal">{v}</div>
                    </div>
                  ))}
                </div>
              </section>
            )}

            <section className="section">
              <h3>Details</h3>
              <div className="accordion">
                <AccordionRow
                  title="Indications"
                  open={openSections.indications}
                  onToggle={() => toggleSection("indications")}
                  preview={clampText(match.indications, 120)}
                  full={match.indications}
                />
                <AccordionRow
                  title="Dosage & Administration"
                  open={openSections.dosage}
                  onToggle={() => toggleSection("dosage")}
                  preview={clampText(match.dosage_and_administration, 120)}
                  full={match.dosage_and_administration}
                />
                <AccordionRow
                  title="Warnings"
                  open={openSections.warnings}
                  onToggle={() => toggleSection("warnings")}
                  preview={clampText(match.warnings, 120)}
                  full={match.warnings}
                />
                <AccordionRow
                  title="Contraindications"
                  open={openSections.contraindications}
                  onToggle={() => toggleSection("contraindications")}
                  preview={clampText(match.contraindications, 120)}
                  full={match.contraindications}
                />
              </div>

              <button
                className="rawToggle"
                onClick={() => toggleSection("raw")}
                type="button"
              >
                {openSections.raw ? "▼ Hide raw JSON" : "▶ Show raw JSON"}
              </button>

              {openSections.raw && (
                <pre className="codeBlock">
                  {JSON.stringify(response, null, 2)}
                </pre>
              )}
            </section>
          </div>
        )}

        <footer className="footer">
          ⚠️ Educational demo only — not a substitute for medical advice
        </footer>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Small components */
/* ------------------------------------------------------------------ */
function Info({ label, value }) {
  return (
    <div className="infoBox">
      <div className="label">{label}</div>
      <div className="value">{value || "—"}</div>
    </div>
  );
}

function AccordionRow({ title, open, onToggle, preview, full }) {
  const hasContent = Boolean(full && full.trim());
  return (
    <div className="accordionRow">
      <div className="accordionHeader">
        <div className="accordionTitle">{title}</div>
        <button
          className="btn btnMini"
          onClick={onToggle}
          disabled={!hasContent}
          type="button"
        >
          {open ? "Hide" : "Show"}
        </button>
      </div>
      <div className="accordionBody">
        {!hasContent ? (
          <div className="muted">—</div>
        ) : open ? (
          <div className="textBlock">{full}</div>
        ) : (
          <div className="textBlock muted">{preview}</div>
        )}
      </div>
    </div>
  );
}
