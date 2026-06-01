/* Improve by Yasiru */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import "./App.css";

// ─── Markdown renderer ────────────────────────────────────────────────────────
// Handles the subset of markdown that the LLM produces:
//   **bold**, *italic*, • / - bullet lines, --- dividers, # headings
function renderInline(text) {
  const parts = text.split(/(\*\*[^*\n]+\*\*|\*[^*\n]+\*)/);
  return parts.map((part, i) => {
    if (part.startsWith("**") && part.endsWith("**"))
      return <strong key={i}>{part.slice(2, -2)}</strong>;
    if (part.startsWith("*") && part.endsWith("*"))
      return <em key={i}>{part.slice(1, -1)}</em>;
    return part || null;
  });
}

function MessageContent({ text }) {
  const lines = (text || "").split("\n");
  const out = [];
  for (let i = 0; i < lines.length; i++) {
    const trimmed = lines[i].trimStart();
    if (/^---+$/.test(trimmed)) {
      out.push(<hr key={i} className="msgDivider" />);
    } else if (/^#{1,3}\s/.test(trimmed)) {
      out.push(<p key={i} className="msgHeading">{renderInline(trimmed.replace(/^#{1,3}\s/, ""))}</p>);
    } else if (/^[•\-\*]\s?/.test(trimmed) && trimmed.length > 1) {
      // Match: •text, - text, * text (with or without space after bullet)
      const content = trimmed.replace(/^[•\-\*]\s*/, "");
      out.push(
        <div key={i} className="msgBullet">
          <span className="msgBulletDot">•</span>
          <span>{renderInline(content)}</span>
        </div>
      );
    } else if (trimmed === "") {
      out.push(<div key={i} className="msgBlank" />);
    } else {
      out.push(<p key={i} className="msgLine">{renderInline(trimmed)}</p>);
    }
  }
  return <div className="msgContent">{out}</div>;
}

const API_BASE = import.meta.env.VITE_API_BASE || "/api";

const DISEASE_OPTIONS = [
  { value: "diabetes", label: "Diabetes" },
  { value: "hypertension", label: "Hypertension" },
  { value: "asthma", label: "Asthma" },
  { value: "heart disease", label: "Heart Disease" },
  { value: "arthritis", label: "Arthritis" },
  { value: "migraine", label: "Migraine" },
];

const LS_KEYS = {
  disease: "ma_disease",
  age: "ma_age",
};

function normalizeDisease(value) {
  return String(value || "")
    .trim()
    .toLowerCase()
    .replace(/\s+/g, " ");
}

function isValidAge(value) {
  const n = Number(value);
  return Number.isInteger(n) && n >= 1 && n <= 120;
}

function clampText(text = "", maxLen = 180) {
  if (!text) return "";
  return text.length <= maxLen ? text : `${text.slice(0, maxLen).trimEnd()}...`;
}

function normalizeSuggestionItem(item) {
  if (typeof item === "string") return item.trim();
  if (item && typeof item === "object") {
    return String(
      item.name ??
        item.label ??
        item.value ??
        item.drug ??
        item.generic_name ??
        "",
    ).trim();
  }
  return "";
}

function statusToUi(status) {
  if (status === "ok") return "ok";
  if (status === "low_confidence") return "low";
  if (status === "no_match") return "none";
  return "idle";
}

function badgeTone(score) {
  const s = Number(score ?? 0);
  if (s >= 95) return "good";
  if (s >= 80) return "warn";
  return "bad";
}

function buildLookupUrl(query, disease, age) {
  return `${API_BASE}/lookup?drug=${encodeURIComponent(query)}&disease=${encodeURIComponent(
    disease,
  )}&age=${encodeURIComponent(age)}`;
}

async function fetchJson(url, options = {}, timeoutMs = 15000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    });
    const contentType = response.headers.get("content-type") || "";
    const payload = contentType.includes("application/json")
      ? await response.json()
      : await response.text();

    if (!response.ok) {
      const message =
        payload?.error?.message ||
        payload?.detail?.message ||
        (typeof payload === "string" && payload) ||
        `API error ${response.status}`;
      throw new Error(message);
    }

    return payload;
  } catch (error) {
    if (error?.name === "AbortError") {
      throw new Error("Request timed out. Check that the backend is running.");
    }
    throw error;
  } finally {
    clearTimeout(timer);
  }
}

export default function App() {
  const [activeView, setActiveView] = useState("lookup");
  const [disease, setDisease] = useState("");
  const [age, setAge] = useState("");
  const [backendStatus, setBackendStatus] = useState("checking");
  const [systemStatus, setSystemStatus] = useState({
    dashboard: null,
  });

  const [drug, setDrug] = useState("");
  const [lookupLoading, setLookupLoading] = useState(false);
  const [lookupResponse, setLookupResponse] = useState(null);
  const [lookupError, setLookupError] = useState("");
  const [showBrandsFull, setShowBrandsFull] = useState(false);
  const [openSections, setOpenSections] = useState({
    indications: false,
    dosage: false,
    warnings: false,
    contraindications: false,
    raw: false,
  });

  const [chatInput, setChatInput] = useState("");
  const [chatDrug, setChatDrug] = useState("");
  const [chatMessages, setChatMessages] = useState([]);
  const [chatLoading, setChatLoading] = useState(false);
  const [chatError, setChatError] = useState("");

  const [systemPanelOpen, setSystemPanelOpen] = useState(false);

  const [uploadFile, setUploadFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [ocrLoading, setOcrLoading] = useState(false);
  const [ocrAnalyzeLoading, setOcrAnalyzeLoading] = useState(false);
  const [ocrResult, setOcrResult] = useState(null);
  const [ocrText, setOcrText] = useState("");
  const [ocrError, setOcrError] = useState("");

  const fileInputRef = useRef(null);

  useEffect(() => {
    localStorage.setItem(LS_KEYS.disease, disease || "");
  }, [disease]);

  useEffect(() => {
    localStorage.setItem(LS_KEYS.age, age || "");
  }, [age]);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  useEffect(() => {
    let cancelled = false;

    const checkBackend = async () => {
      try {
        await fetchJson(`${API_BASE}/health`, {}, 8000);
        if (!cancelled) setBackendStatus("online");
      } catch (error) {
        if (!cancelled) setBackendStatus("offline");
      }
    };

    const loadSystemStatus = async () => {
      try {
        const dashboard = await fetchJson(`${API_BASE}/dashboard`, {}, 10000);

        if (!cancelled) {
          setSystemStatus({ dashboard });
        }
      } catch (error) {
        if (!cancelled) {
          setSystemStatus({ dashboard: null });
        }
      }
    };

    checkBackend();
    loadSystemStatus();

    return () => {
      cancelled = true;
    };
  }, []);

  const contextReady = disease.trim() && isValidAge(age);
  const contextMessage = useMemo(() => {
    if (!disease.trim()) return "Select a disease.";
    if (!isValidAge(age)) return "Enter age 1-120.";
    return "";
  }, [disease, age]);

  const lookupStatus = statusToUi(lookupResponse?.status);
  const match = lookupResponse?.best_match || null;
  const bestScore = Number(lookupResponse?.best_score ?? match?.score ?? 0);
  const confidencePercent = Number.isFinite(Number(lookupResponse?.confidence))
    ? Math.round(Number(lookupResponse?.confidence) * 100)
    : null;
  const suggestions = useMemo(() => {
    const raw = lookupResponse?.suggestions;
    if (!Array.isArray(raw)) return [];
    return Array.from(
      new Set(raw.map(normalizeSuggestionItem).filter(Boolean)),
    );
  }, [lookupResponse]);

  const anyLoading =
    lookupLoading || chatLoading || ocrLoading || ocrAnalyzeLoading;

  const featureStatus = useMemo(() => {
    const dashboard = systemStatus.dashboard || {};
    const meta = dashboard.meta || {};
    const memory = dashboard.memory || {};
    const activity = dashboard.activity || {};
    const counts = activity.counts || {};
    const ocrTotal = (counts.ocr || 0) + (counts.ocr_text || 0);

    return [
      {
        label: "Backend",
        tone: backendStatus,
        detail:
          backendStatus === "online"
            ? "API responding"
            : backendStatus === "offline"
              ? "No API connection"
              : "Checking",
      },
      {
        label: "LLM",
        tone: dashboard.llm_available ? "online" : "offline",
        detail: dashboard.llm_available
          ? "Grounded chat enabled"
          : "Rule-based fallback",
      },
      {
        label: "RAG / FAISS",
        tone: dashboard.faiss?.index_ready ? "online" : "offline",
        detail: dashboard.faiss?.index_ready
          ? `${dashboard.faiss?.vector_count || 0} vectors indexed`
          : "Vector index not loaded",
      },
      {
        label: "OCR",
        tone: meta.features?.prescription_ocr ? "online" : "offline",
        detail: dashboard.ocr_runtime?.available
          ? "Image OCR ready"
          : "OCR fallback mode",
      },
      {
        label: "NER",
        tone: meta.features?.lightweight_ner ? "online" : "offline",
        detail: dashboard.ner?.mode || "Unknown",
      },
      {
        label: "Lookups",
        tone: counts.lookup > 0 ? "online" : "offline",
        detail: `${counts.lookup || 0} recorded`,
      },
      {
        label: "Chats",
        tone: counts.chat > 0 ? "online" : "offline",
        detail: `${counts.chat || 0} recorded`,
      },
      {
        label: "OCR jobs",
        tone: ocrTotal > 0 ? "online" : "offline",
        detail: `${ocrTotal} recorded`,
      },
    ];
  }, [backendStatus, systemStatus]);

  const recentActivity = systemStatus.dashboard?.activity?.recent_events || [];

  const resetLookupToggles = () => {
    setShowBrandsFull(false);
    setOpenSections({
      indications: false,
      dosage: false,
      warnings: false,
      contraindications: false,
      raw: false,
    });
  };

  const lookupDrugWithValue = async (value) => {
    const query = String(value || "").trim();
    const d = normalizeDisease(disease);
    const a = Number(age);

    if (!d || !isValidAge(a)) {
      setLookupError(contextMessage || "Set disease and age first.");
      return;
    }
    if (!query) {
      setLookupError("Enter a drug name.");
      return;
    }

    setLookupLoading(true);
    setLookupError("");
    setLookupResponse(null);
    resetLookupToggles();

    try {
      const data = await fetchJson(buildLookupUrl(query, d, a));
      setLookupResponse(data);
    } catch (error) {
      setLookupError(error?.message || "Lookup failed.");
    } finally {
      setLookupLoading(false);
    }
  };

  const runLookup = () => lookupDrugWithValue(drug);

  const applySuggestion = (suggestion) => {
    setDrug(suggestion);
    lookupDrugWithValue(suggestion);
  };

  const sendChat = async (messageOverride, drugOverride) => {
    const message = String(messageOverride ?? chatInput).trim();
    const d = normalizeDisease(disease);
    const a = Number(age);
    const selectedDrug = String(drugOverride ?? chatDrug).trim();

    if (!d || !isValidAge(a)) {
      setChatError(contextMessage || "Set disease and age first.");
      return;
    }
    if (!message) {
      setChatError("Enter a message.");
      return;
    }

    setChatLoading(true);
    setChatError("");
    setChatInput("");
    setChatMessages((items) => [...items, { role: "user", text: message }]);

    const payload = { message, disease: d, age: a };
    if (selectedDrug) payload.drug = selectedDrug;

    try {
      const data = await fetchJson(
        `${API_BASE}/chat`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        },
        45000,  // 45 s — Groq LLM + FAISS search can take 10-15 s; first call loads embeddings
      );
      setChatMessages((items) => [
        ...items,
        { role: "assistant", text: data.answer, data },
      ]);
    } catch (error) {
      setChatError(error?.message || "Chat failed.");
    } finally {
      setChatLoading(false);
    }
  };

  const useLookupInChat = () => {
    const name = match?.generic_name_clean || match?.generic_name || drug;
    if (!name) return;
    setChatDrug(name);
    setChatInput(`Is ${name} safe for my context?`);
    setActiveView("chat");
  };

  const onFileChange = (event) => {
    const file = event.target.files?.[0] || null;
    setUploadFile(file);
    setOcrResult(null);
    setOcrText("");
    setOcrError("");

    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(file ? URL.createObjectURL(file) : "");
  };

  const uploadPrescription = async () => {
    const d = normalizeDisease(disease);
    const a = Number(age);

    if (!d || !isValidAge(a)) {
      setOcrError(contextMessage || "Set disease and age first.");
      return;
    }
    if (!uploadFile) {
      setOcrError("Choose an image file.");
      return;
    }

    const form = new FormData();
    form.append("file", uploadFile);
    form.append("disease", d);
    form.append("age", String(a));

    setOcrLoading(true);
    setOcrError("");
    setOcrResult(null);
    setOcrText("");

    try {
      const data = await fetchJson(
        `${API_BASE}/prescription`,
        { method: "POST", body: form },
        45000,
      );
      setOcrResult(data);
      setOcrText(data.ocr?.text || "");
    } catch (error) {
      setOcrError(error?.message || "Prescription OCR failed.");
    } finally {
      setOcrLoading(false);
    }
  };

  const analyzeOcrText = async () => {
    const d = normalizeDisease(disease);
    const a = Number(age);
    const text = String(ocrText || "").trim();

    if (!d || !isValidAge(a)) {
      setOcrError(contextMessage || "Set disease and age first.");
      return;
    }
    if (!text) {
      setOcrError("Enter OCR text to analyze.");
      return;
    }

    setOcrAnalyzeLoading(true);
    setOcrError("");

    try {
      const data = await fetchJson(
        `${API_BASE}/prescription/analyze-text`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text, disease: d, age: a }),
        },
        20000,
      );
      setOcrResult((previous) => ({
        ...(previous || {}),
        ...data,
        file: previous?.file,
        preprocessing: previous?.preprocessing,
      }));
      setOcrText(data.ocr?.text || text);
    } catch (error) {
      setOcrError(error?.message || "Text analysis failed.");
    } finally {
      setOcrAnalyzeLoading(false);
    }
  };

  const clearChat = useCallback(() => {
    setChatMessages([]);
    setChatInput("");
    setChatDrug("");
    setChatError("");
  }, []);

  const sendDetectedToChat = (medicine) => {
    const name =
      medicine?.drug || medicine?.normalized || medicine?.query || "";
    if (!name) return;
    setActiveView("chat");
    setChatDrug(name);
    sendChat(`Explain ${name} from this prescription for my context.`, name);
  };

  const sendAllToChat = useCallback(
    (medicines) => {
      if (!medicines?.length) return;
      const names = medicines
        .map((m) => m.drug || m.normalized || m.query)
        .filter(Boolean);
      if (!names.length) return;
      setChatDrug(names[0]);
      setChatInput(
        `My prescription contains: ${names.join(", ")}. Give me an overview of each medicine.`
      );
      setActiveView("chat");
    },
    []
  );

  return (
    <main className="appShell">
      <section className="workspace">
        {/* ── App header ── */}
        <header className="topBar">
          <div className="topBarBrand">
            <div className="eyebrow">MediAssist Bot</div>
            <h1>Medication assistant</h1>
          </div>
          <div className={`backendDot ${backendStatus}`} title={backendStatus === "online" ? "Backend online" : "Backend offline"} />
        </header>

        {/* ── Patient context (3-column compact) ── */}
        <section className="contextPanel" aria-label="Patient context">
          <label className="contextField">
            <span>Condition</span>
            <select value={disease} onChange={(e) => setDisease(normalizeDisease(e.target.value))}>
              <option value="">Select…</option>
              {DISEASE_OPTIONS.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
            </select>
          </label>
          <label className="contextField">
            <span>Age</span>
            <input type="number" min="1" max="120" inputMode="numeric" value={age}
              onChange={(e) => setAge(e.target.value)} placeholder="1–120" />
          </label>
          <div className={`contextBadge ${contextReady ? "ready" : "needs"}`}>
            {contextReady ? "✓ Ready" : contextMessage}
          </div>
        </section>

        {/* ── System status (collapsible) ── */}
        <section className="systemPanel" aria-label="System status">
          <button className="systemPanelToggle" type="button"
            onClick={() => setSystemPanelOpen((v) => !v)}>
            <div className="systemPanelToggleLeft">
              <span className={`backendDot ${backendStatus}`} />
              <span className="systemPanelTitle">System status</span>
              <div className="statusMiniRow">
                {featureStatus.slice(0, 4).map((s) => (
                  <span key={s.label} className={`miniPill ${s.tone}`}>{s.label}</span>
                ))}
              </div>
            </div>
            <span className="systemPanelChevron">{systemPanelOpen ? "▲" : "▼"}</span>
          </button>

          {systemPanelOpen && (
            <div className="systemPanelBody">
              <div className="statusGrid">
                {featureStatus.map((item) => (
                  <div key={item.label} className="statusCard">
                    <div className="statusCardTop">
                      <strong>{item.label}</strong>
                      <span className={`statusPill ${item.tone}`}>{item.tone}</span>
                    </div>
                    <p>{item.detail}</p>
                  </div>
                ))}
              </div>

              <div className="statusDivider" />

              <div className="activityWrap">
                <div className="sectionHead compact">
                  <h3>Recent activity</h3>
                  <span className="smallCaps">{recentActivity.length} events</span>
                </div>
                {recentActivity.length ? (
                  <div className="activityList">
                    {recentActivity.slice(0, 4).map((event, index) => (
                      <div key={`${event.timestamp}-${index}`} className="activityItem">
                        <div className="activityItemTop">
                          <strong>{event.kind}</strong>
                          <span className={`statusPill ${event.success ? "online" : "offline"}`}>
                            {event.success ? "ok" : "err"}
                          </span>
                        </div>
                        <p>{event.detail || "Recorded event"}</p>
                        <div className="activityMeta">
                          <span>{event.timestamp}</span>
                          <span>{event.status_code || "-"}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="muted" style={{padding:"8px 0"}}>No activity yet.</p>
                )}
              </div>
            </div>
          )}
        </section>

        {/* ── Tabs ── */}
        <nav className="tabs" aria-label="MediAssist views">
          {[
            ["lookup", "🔍", "Lookup"],
            ["chat",   "💬", "Chat"],
            ["prescription", "📋", "Prescription"],
          ].map(([id, icon, label]) => (
            <button key={id} type="button"
              className={activeView === id ? "tab active" : "tab"}
              onClick={() => setActiveView(id)}>
              <span className="tabIcon">{icon}</span>
              <span>{label}</span>
            </button>
          ))}
        </nav>

        {activeView === "lookup" && (
          <LookupView
            drug={drug}
            setDrug={setDrug}
            loading={lookupLoading}
            error={lookupError}
            response={lookupResponse}
            status={lookupStatus}
            match={match}
            bestScore={bestScore}
            confidencePercent={confidencePercent}
            suggestions={suggestions}
            showBrandsFull={showBrandsFull}
            setShowBrandsFull={setShowBrandsFull}
            openSections={openSections}
            setOpenSections={setOpenSections}
            onLookup={runLookup}
            onSuggestion={applySuggestion}
            onUseInChat={useLookupInChat}
            contextReady={contextReady}
          />
        )}

        {activeView === "chat" && (
          <ChatView
            chatInput={chatInput}
            setChatInput={setChatInput}
            chatDrug={chatDrug}
            setChatDrug={setChatDrug}
            messages={chatMessages}
            loading={chatLoading}
            error={chatError}
            onSend={() => sendChat()}
            onClear={clearChat}
            contextReady={contextReady}
          />
        )}

        {activeView === "prescription" && (
          <PrescriptionView
            fileInputRef={fileInputRef}
            uploadFile={uploadFile}
            previewUrl={previewUrl}
            loading={ocrLoading}
            analyzeLoading={ocrAnalyzeLoading}
            error={ocrError}
            result={ocrResult}
            ocrText={ocrText}
            setOcrText={setOcrText}
            onFileChange={onFileChange}
            onUpload={uploadPrescription}
            onAnalyzeText={analyzeOcrText}
            onSendToChat={sendDetectedToChat}
            onSendAllToChat={sendAllToChat}
            contextReady={contextReady}
          />
        )}

        <footer className="footer">
          Educational demo only · Confirm all medication decisions with a licensed clinician
        </footer>
      </section>
    </main>
  );
}

function LookupView({
  drug,
  setDrug,
  loading,
  error,
  response,
  status,
  match,
  bestScore,
  confidencePercent,
  suggestions,
  showBrandsFull,
  setShowBrandsFull,
  openSections,
  setOpenSections,
  onLookup,
  onSuggestion,
  onUseInChat,
  contextReady,
}) {
  const tone = badgeTone(bestScore);
  const brandNames = match?.brand_names || "";
  const brandShort = clampText(brandNames, 170);

  const toggleSection = (key) => {
    setOpenSections((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  return (
    <section className="viewStack">
      {/* ── Unified search bar ── */}
      <div className="searchBar">
        <span className="searchIcon">🔍</span>
        <input
          className="searchInput"
          value={drug}
          onChange={(event) => setDrug(event.target.value)}
          onKeyDown={(event) => { if (event.key === "Enter") onLookup(); }}
          placeholder="Enter drug name or brand (e.g. Metformin, Aspirin)"
          autoComplete="off"
          spellCheck="false"
        />
        <button
          className="searchBtn"
          type="button"
          onClick={onLookup}
          disabled={loading || !contextReady}
        >
          {loading ? "Searching…" : "Search"}
        </button>
      </div>

      {error && <Notice tone="bad">{error}</Notice>}

      {response && status === "none" && (
        <Notice tone="bad">No confident match found.</Notice>
      )}
      {response && status === "low" && (
        <Notice tone="warn">Low confidence match. Check suggestions.</Notice>
      )}

      {response &&
        (status === "low" || status === "none") &&
        suggestions.length > 0 && (
          <section className="panel">
            <div className="sectionHead">
              <h2>Suggestions</h2>
              <span>{suggestions.length}</span>
            </div>
            <div className="suggestGrid">
              {suggestions.map((suggestion) => (
                <button
                  key={suggestion}
                  type="button"
                  className="suggestButton"
                  onClick={() => onSuggestion(suggestion)}
                >
                  <span>{suggestion}</span>
                  <span>Use</span>
                </button>
              ))}
            </div>
          </section>
        )}

      {response && match && status !== "none" && (
        <section className="panel">
          <div className="resultTop">
            <div>
              <h2>{match.generic_name_clean || match.generic_name}</h2>
              <p>Matched as {match.match}</p>
            </div>
            <div className={`matchBadge ${tone}`}>
              {confidencePercent ?? Math.round(bestScore)}% confidence
            </div>
          </div>

          {response.context_highlights?.highlights && (
            <ContextHighlights data={response.context_highlights} />
          )}

          <div className="infoGrid">
            <Info label="Route" value={match.route} />
            <Info label="Drug class" value={match.drug_class} />
            <Info
              label="Sources"
              value={(match.sources || "-").replaceAll("|", " | ")}
            />
            <Info label="Last updated" value={match.last_updated || "-"} />
            <div className="infoItem wide">
              <span>Brand names</span>
              <strong>
                {brandNames ? (showBrandsFull ? brandNames : brandShort) : "-"}
              </strong>
              {brandNames.length > 170 && (
                <button
                  className="inlineButton"
                  type="button"
                  onClick={() => setShowBrandsFull((value) => !value)}
                >
                  {showBrandsFull ? "Show less" : "Show more"}
                </button>
              )}
            </div>
          </div>

          {match.side_effects_buckets && (
            <section className="contentBlock">
              <h3>Side Effects</h3>
              <div className="sideEffectGrid">
                {Object.entries(match.side_effects_buckets).map(
                  ([key, value]) => (
                    <div className="sideEffectRow" key={key}>
                      <span>{key.replaceAll("_", " ")}</span>
                      <p>{value}</p>
                    </div>
                  ),
                )}
              </div>
            </section>
          )}

          <section className="contentBlock">
            <h3>Details</h3>
            <div className="accordion">
              <AccordionRow
                title="Indications"
                open={openSections.indications}
                onToggle={() => toggleSection("indications")}
                preview={clampText(match.indications, 140)}
                full={match.indications}
              />
              <AccordionRow
                title="Dosage and Administration"
                open={openSections.dosage}
                onToggle={() => toggleSection("dosage")}
                preview={clampText(match.dosage_and_administration, 140)}
                full={match.dosage_and_administration}
              />
              <AccordionRow
                title="Warnings"
                open={openSections.warnings}
                onToggle={() => toggleSection("warnings")}
                preview={clampText(match.warnings, 140)}
                full={match.warnings}
              />
              <AccordionRow
                title="Contraindications"
                open={openSections.contraindications}
                onToggle={() => toggleSection("contraindications")}
                preview={clampText(match.contraindications, 140)}
                full={match.contraindications}
              />
            </div>
          </section>

          <div className="actionsRow">
            <button
              className="secondaryButton"
              type="button"
              onClick={onUseInChat}
            >
              Send to chat
            </button>
            <button
              className="ghostButton"
              type="button"
              onClick={() => toggleSection("raw")}
            >
              {openSections.raw ? "Hide JSON" : "Show JSON"}
            </button>
          </div>

          {openSections.raw && (
            <pre className="codeBlock">{JSON.stringify(response, null, 2)}</pre>
          )}
        </section>
      )}

      {/* empty state when nothing searched yet */}
      {!response && !error && (
        <div className="lookupEmptyState">
          <div className="lookupEmptyIcon">💊</div>
          <p className="lookupEmptyTitle">Search the drug knowledge base</p>
          <p className="muted">Type a generic name, brand name, or partial spelling — fuzzy matching handles typos.</p>
          <div className="lookupEmptyHints">
            {["Metformin", "Lisinopril", "Salbutamol", "Aspirin", "Sumatriptan"].map((hint) => (
              <button key={hint} type="button" className="hintChip"
                onClick={() => { setDrug(hint); onLookup(); }}
                disabled={!contextReady}
              >{hint}</button>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}

function ChatView({
  chatInput,
  setChatInput,
  chatDrug,
  setChatDrug,
  messages,
  loading,
  error,
  onSend,
  onClear,
  contextReady,
}) {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  return (
    <section className="viewStack">
      <div className="chatPanel">
        {/* ── Header ── */}
        <div className="chatHeader">
          <span className="chatHeaderLabel">Chat</span>
          {messages.length > 0 && (
            <button className="chatClearBtn" type="button" onClick={onClear}>
              Clear chat
            </button>
          )}
        </div>

        {/* ── Messages ── */}
        <div className="chatMessages">
          {messages.length === 0 && (
            <div className="chatEmptyState">
              <p className="chatEmptyTitle">Ask about your medication</p>
              <p className="muted">Try: "What are the side effects of metformin?" or upload a prescription first.</p>
            </div>
          )}
          {messages.map((message, index) => {
            const isEmergency = message.data?.is_emergency;
            const cls = `message ${message.role}${isEmergency ? " emergency" : ""}`;
            return (
              <div key={`${message.role}-${index}`} className={cls}>
                {message.role === "assistant" ? (
                  <MessageContent text={message.text} />
                ) : (
                  <p className="msgLine">{message.text}</p>
                )}
                {message.data?.matched_drugs?.length > 0 && (
                  <MatchedDrugStrip items={message.data.matched_drugs} />
                )}
                {message.data?.answer_source && (
                  <span className="msgSource">
                    {message.data.answer_source === "llm_grounded" ? "LLM" : "Rule-based"}
                  </span>
                )}
              </div>
            );
          })}
          {loading && (
            <div className="message assistant loading">
              <span className="typingDot" /><span className="typingDot" /><span className="typingDot" />
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        {/* ── Composer ── */}
        <div className="chatComposer">
          <input
            className="textInput chatDrugInput"
            value={chatDrug}
            onChange={(event) => setChatDrug(event.target.value)}
            placeholder="Drug name (optional)"
          />
          <div className="chatInputRow">
            <textarea
              value={chatInput}
              onChange={(event) => setChatInput(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  onSend();
                }
              }}
              placeholder="Ask about dosage, side effects, interactions… (Enter to send)"
            />
            <button
              className="primaryButton chatSendBtn"
              type="button"
              onClick={onSend}
              disabled={loading || !contextReady}
            >
              {loading ? "…" : "Send"}
            </button>
          </div>
        </div>
      </div>
      {error && <Notice tone="bad">{error}</Notice>}
    </section>
  );
}

function PrescriptionView({
  fileInputRef,
  uploadFile,
  previewUrl,
  loading,
  analyzeLoading,
  error,
  result,
  ocrText,
  setOcrText,
  onFileChange,
  onUpload,
  onAnalyzeText,
  onSendToChat,
  onSendAllToChat,
  contextReady,
}) {
  const medicines = result?.detected_medicines || [];
  const ocrConfidence = result?.ocr?.confidence ?? null;
  const confidencePercent =
    ocrConfidence !== null ? Math.round(ocrConfidence * 100) : null;

  const getConfidenceTone = (pct) => {
    if (pct === null) return "none";
    if (pct >= 85) return "high";
    if (pct >= 70) return "medium";
    return "low";
  };

  const confidenceTone = getConfidenceTone(confidencePercent);

  return (
    <section className="viewStack">
      <section className="uploadGrid">
        <div className="uploadPanel">
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={onFileChange}
          />
          <button
            className="primaryButton"
            type="button"
            onClick={onUpload}
            disabled={loading || !contextReady || !uploadFile}
          >
            {loading ? "Extracting..." : "Extract text"}
          </button>
          {uploadFile && <div className="fileName">📄 {uploadFile.name}</div>}
          {!uploadFile && (
            <div className="emptyState">Choose a prescription image</div>
          )}
        </div>

        <div className="previewPanel">
          {previewUrl ? (
            <img src={previewUrl} alt="Prescription preview" />
          ) : (
            <div className="emptyState">No image selected</div>
          )}
        </div>
      </section>

      {error && <Notice tone="bad">{error}</Notice>}

      {/* Text editor — always visible so users can paste text directly */}
      <section className="panel">
        {/* OCR confidence bar — only after image extraction */}
        {result && confidencePercent !== null && (
          <div className={`ocrConfidenceBanner ${confidenceTone}`}>
            <div className="ocrConfidenceTop">
              <span className="ocrConfidenceLabel">OCR Confidence</span>
              <span className="ocrConfidenceScore">{confidencePercent}%</span>
            </div>
            <div className="ocrConfidenceTrack">
              <div
                className="ocrConfidenceFill"
                style={{ width: `${Math.min(confidencePercent, 100)}%` }}
              />
            </div>
            <p className="ocrConfidenceHint">
              {confidenceTone === "high" && "✓ Text is highly reliable — proceed to detect medicines."}
              {confidenceTone === "medium" && "⚡ Looks good — review a few words before re-detecting."}
              {confidenceTone === "low" && "⚠ Low confidence — correct the text above before analyzing."}
            </p>
          </div>
        )}

        <div className="ocrEditor">
          <label>
            <span>{result ? "Extracted or corrected text" : "Paste prescription text"}</span>
            <textarea
              value={ocrText}
              onChange={(event) => setOcrText(event.target.value)}
              placeholder={
                result
                  ? "OCR text appears here. Edit if needed, then click Re-detect medicines."
                  : "Paste prescription text here to detect medicines — no image needed."
              }
            />
            <div className="ocrEditorHint">
              {result
                ? "✏️ Edit above and click \"Re-detect medicines\" to update results."
                : "✏️ Or upload a prescription image above to auto-fill this text."}
            </div>
          </label>
          <button
            className="secondaryButton"
            type="button"
            onClick={onAnalyzeText}
            disabled={analyzeLoading || !contextReady || !ocrText.trim()}
          >
            {analyzeLoading ? "Analyzing..." : result ? "Re-detect medicines" : "Detect medicines"}
          </button>
        </div>

        {/* Detected medicines — shown once analysis has run */}
        {result && (
          <section className="contentBlock">
            <div className="sectionHead">
              <h3>Detected Medicines</h3>
              <div className="detectedActions">
                {medicines.length > 0 && (
                  <span className="smallCaps">{medicines.length} found</span>
                )}
                {medicines.length > 1 && (
                  <button
                    className="secondaryButton detectedSendAllBtn"
                    type="button"
                    onClick={() => onSendAllToChat(medicines)}
                  >
                    Chat about all {medicines.length}
                  </button>
                )}
              </div>
            </div>
            {medicines.length ? (
              <div className="detectedList">
                {medicines.map((medicine, index) => (
                  <div
                    className="detectedItem"
                    key={`${medicine.drug}-${index}`}
                  >
                    <div className="detectedItemInfo">
                      <strong className="detectedDrugName">
                        {medicine.drug || medicine.normalized || medicine.query}
                      </strong>
                      <span className="detectedDrugMeta">
                        {[medicine.dosage, medicine.frequency, medicine.route]
                          .filter(Boolean)
                          .join(" · ") || "No dosage detected"}
                      </span>
                    </div>
                    <button
                      className="secondaryButton detectedChatBtn"
                      type="button"
                      onClick={() => onSendToChat(medicine)}
                    >
                      Chat
                    </button>
                  </div>
                ))}
              </div>
            ) : (
              <div className="emptyState">
                No medicines detected.
                <br />
                Try a clearer image or check the prescription text above.
              </div>
            )}
          </section>
        )}
      </section>
    </section>
  );
}

function ContextHighlights({ data }) {
  const entries = Object.entries(data.highlights || {});
  if (!entries.length) return null;

  return (
    <section className="highlightBox">
      <div className="highlightMeta">Age group: {data.age_group || "-"}</div>
      {entries.map(([key, values]) => (
        <div key={key}>
          <strong>{key.replaceAll("_", " ")}</strong>
          <ul>
            {(Array.isArray(values) ? values : [values]).map((value, index) => (
              <li key={`${key}-${index}`}>{value}</li>
            ))}
          </ul>
        </div>
      ))}
    </section>
  );
}

function MatchedDrugStrip({ items }) {
  return (
    <div className="drugStrip">
      {items
        .filter((item) => item.best_match)
        .slice(0, 3)
        .map((item) => (
          <span key={`${item.query}-${item.normalized}`}>
            {item.best_match?.generic_name_clean ||
              item.best_match?.generic_name ||
              item.query}{" "}
            - {Math.round(Number(item.confidence || 0) * 100)}%
          </span>
        ))}
    </div>
  );
}

function Info({ label, value }) {
  return (
    <div className="infoItem">
      <span>{label}</span>
      <strong>{value || "-"}</strong>
    </div>
  );
}

function AccordionRow({ title, open, onToggle, preview, full }) {
  const hasContent = Boolean(full && full.trim());
  return (
    <div className="accordionRow">
      <button type="button" onClick={onToggle} disabled={!hasContent}>
        <span>{title}</span>
        <span>{open ? "Hide" : "Show"}</span>
      </button>
      <div className={open ? "accordionBody open" : "accordionBody"}>
        {hasContent ? (open ? full : preview) : "-"}
      </div>
    </div>
  );
}

function Notice({ tone, children }) {
  return <div className={`notice ${tone}`}>{children}</div>;
}
