import { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";

const API_BASE = import.meta.env.VITE_API_BASE || "/api";

const DISEASE_OPTIONS = [
  { value: "diabetes", label: "Diabetes" },
  { value: "hypertension", label: "Hypertension" },
  { value: "asthma", label: "Asthma" },
  { value: "heart disease", label: "Heart Disease" },
  { value: "arthritis", label: "Arthritis" },
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
  const [disease, _setDisease] = useState(() =>
    normalizeDisease(localStorage.getItem(LS_KEYS.disease)),
  );
  const [age, _setAge] = useState(
    () => localStorage.getItem(LS_KEYS.age) || "",
  );

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
        20000,
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

  const sendDetectedToChat = (medicine) => {
    const name =
      medicine?.drug || medicine?.normalized || medicine?.query || "";
    if (!name) return;
    setActiveView("chat");
    setChatDrug(name);
    sendChat(`Explain ${name} from this prescription for my context.`, name);
  };

  return (
    <main className="appShell">
      <section className="workspace">
        <header className="topBar">
          <div>
            <div className="eyebrow">MediAssist Bot</div>
            <h1>Medication assistant</h1>
          </div>
          <div className="apiBadge">{API_BASE}</div>
        </header>

        <div className="searchRow">
          <input
            className="input"
            placeholder="Enter drug name (e.g. paracetamol)"
            value={drug}
            onChange={(e) => setDrug(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") runLookup();
            }}
          />

          <button
            className="btn btnPrimary"
            onClick={runLookup}
            disabled={lookupLoading || !drug.trim()}
          >
            {lookupLoading ? "Searching…" : "Search"}
          </button>

          <button
            className="btn btnGhost"
            onClick={() => {
              setDrug("");
              setLookupResponse(null);
              setLookupError("");
              resetLookupToggles();
            }}
            disabled={lookupLoading}
          >
            Clear
          </button>
        </div>

        {lookupResponse?.query && (
          <div className="apiLine">
            <span className="muted">API:</span>{" "}
            <a
              href={`${API_BASE}/lookup?drug=${encodeURIComponent(lookupResponse.query)}`}
              target="_blank"
              rel="noreferrer"
            >
              {`${API_BASE}/lookup?drug=${encodeURIComponent(lookupResponse.query)}`}
            </a>
          </div>
        )}

        <nav className="tabs" aria-label="MediAssist views">
          {[
            ["lookup", "Lookup"],
            ["chat", "Chat"],
            ["prescription", "Prescription"],
          ].map(([id, label]) => (
            <button
              key={id}
              type="button"
              className={activeView === id ? "tab active" : "tab"}
              onClick={() => setActiveView(id)}
            >
              {label}
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
            contextReady={contextReady}
          />
        )}

        <footer className="footer">
          Educational demo only. Confirm medication decisions with a licensed
          clinician.
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
      <div className="toolRow">
        <input
          className="textInput"
          value={drug}
          onChange={(event) => setDrug(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") onLookup();
          }}
          placeholder="Drug name or brand"
        />
        <button
          className="primaryButton"
          type="button"
          onClick={onLookup}
          disabled={loading || !contextReady}
        >
          {loading ? "Searching..." : "Search"}
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
  contextReady,
}) {
  return (
    <section className="viewStack">
      <div className="chatPanel">
        <div className="chatMessages">
          {messages.length === 0 && (
            <div className="emptyState">No chat messages yet.</div>
          )}
          {messages.map((message, index) => (
            <div
              key={`${message.role}-${index}`}
              className={`message ${message.role}`}
            >
              <pre>{message.text}</pre>
              {message.data?.matched_drugs?.length > 0 && (
                <MatchedDrugStrip items={message.data.matched_drugs} />
              )}
            </div>
          ))}
          {loading && (
            <div className="message assistant loading">Thinking...</div>
          )}
        </div>

        <div className="chatComposer">
          <input
            className="textInput"
            value={chatDrug}
            onChange={(event) => setChatDrug(event.target.value)}
            placeholder="Optional drug"
          />
          <textarea
            value={chatInput}
            onChange={(event) => setChatInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                onSend();
              }
            }}
            placeholder="Ask a medication question"
          />
          <button
            className="primaryButton"
            type="button"
            onClick={onSend}
            disabled={loading || !contextReady}
          >
            {loading ? "Sending..." : "Send"}
          </button>
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

      {result && (
        <section className="panel">
          <div className="sectionHead">
            <h2>OCR Result</h2>
            {confidencePercent !== null && (
              <span className={`confidenceIndicator ${confidenceTone}`}>
                {confidencePercent}% confidence
              </span>
            )}
          </div>

          {/* OCR Confidence Bar */}
          {confidencePercent !== null && (
            <div className="confidenceBar">
              <div className="confidenceBarLabel">
                <span className="label">OCR Extraction Confidence</span>
                <span className="value">{confidencePercent}%</span>
              </div>
              <div className="confidenceBarTrack">
                <div
                  className={`confidenceBarFill ${confidenceTone}`}
                  style={{ width: `${Math.min(confidencePercent, 100)}%` }}
                />
              </div>
              <div className="ocrEditorHint">
                {confidenceTone === "high" && "OCR text is highly reliable."}
                {confidenceTone === "medium" &&
                  "OCR text looks good but review for accuracy."}
                {confidenceTone === "low" &&
                  "OCR text may have errors. Please review and correct before proceeding."}
              </div>
            </div>
          )}

          <div className="ocrEditor">
            <label>
              <span>Extracted or corrected text</span>
              <textarea
                value={ocrText}
                onChange={(event) => setOcrText(event.target.value)}
                placeholder="OCR text will appear here. Edit if needed before re-detecting medicines."
              />
              <div className="ocrEditorHint">
                ✏️ Edit text above and click "Re-detect medicines" to analyze
                changes
              </div>
            </label>
            <button
              className="secondaryButton"
              type="button"
              onClick={onAnalyzeText}
              disabled={analyzeLoading || !contextReady || !ocrText.trim()}
            >
              {analyzeLoading ? "Analyzing..." : "Re-detect medicines"}
            </button>
          </div>

          <section className="contentBlock">
            <h3>Detected Medicines</h3>
            {medicines.length ? (
              <div className="detectedList">
                {medicines.map((medicine, index) => (
                  <div
                    className="detectedItem"
                    key={`${medicine.drug}-${index}`}
                  >
                    <div>
                      <strong>
                        {medicine.drug || medicine.normalized || medicine.query}
                      </strong>
                      <span>
                        {[medicine.dosage, medicine.frequency, medicine.route]
                          .filter(Boolean)
                          .join(" | ") || "No dosage pattern detected"}
                      </span>
                    </div>
                    <button
                      className="secondaryButton"
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
                No medicines detected in OCR text.
                <br />
                Try uploading a clearer prescription image or manually entering
                medicine names.
              </div>
            )}
          </section>
        </section>
      )}

      {!result && !error && !uploadFile && (
        <div
          className="panel"
          style={{ padding: "32px 16px", textAlign: "center" }}
        >
          <div
            className="emptyState"
            style={{
              minHeight: "200px",
              display: "grid",
              placeItems: "center",
            }}
          >
            <div>
              <h3 style={{ marginBottom: "8px" }}>📸 Upload a Prescription</h3>
              <p style={{ color: "var(--muted)", margin: "0" }}>
                Choose a prescription image (JPG, PNG) to extract medicine
                information automatically.
              </p>
            </div>
          </div>
        </div>
      )}
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
