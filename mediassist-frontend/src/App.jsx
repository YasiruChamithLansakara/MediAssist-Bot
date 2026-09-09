import { useCallback, useEffect, useState } from "react";
import { ApiError, accessCode, api } from "./api.js";
import { Alert, Card, Pill } from "./components/ui.jsx";
import { Icon } from "./components/icons.jsx";
import AccessGate from "./components/AccessGate.jsx";
import Lookup from "./components/Lookup.jsx";
import Chat from "./components/Chat.jsx";
import Prescription from "./components/Prescription.jsx";
import InteractionChecker from "./components/InteractionChecker.jsx";

const VIEWS = [
  { id: "chat", label: "Ask", icon: Icon.Chat, tag: "AI" },
  { id: "lookup", label: "Look up", icon: Icon.Pill },
  { id: "prescription", label: "Prescription", icon: Icon.Scan, tag: "OCR" },
  { id: "interactions", label: "Interactions", icon: Icon.Link, tag: "new" },
];

const THEME_KEY = "mediassist.theme";

function useTheme() {
  const [theme, setTheme] = useState(() => {
    try {
      const stored = localStorage.getItem(THEME_KEY);
      if (stored) return stored;
    } catch {
      /* ignore */
    }
    return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  });

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    try {
      localStorage.setItem(THEME_KEY, theme);
    } catch {
      /* ignore */
    }
  }, [theme]);

  return [theme, () => setTheme((t) => (t === "dark" ? "light" : "dark"))];
}

export default function App() {
  const [theme, toggleTheme] = useTheme();

  const [gateChecked, setGateChecked] = useState(false);
  const [locked, setLocked] = useState(false);

  const [view, setView] = useState("chat");
  const [disease, setDisease] = useState("");
  const [age, setAge] = useState("");
  const [meta, setMeta] = useState(null);
  const [bootError, setBootError] = useState("");

  /* ---------------------------------------------------------------- gate */
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const config = await api.config();
        if (cancelled) return;
        if (!config.access_required) {
          setLocked(false);
        } else if (!accessCode.get()) {
          setLocked(true);
        } else {
          // A stored code may have been rotated since it was saved.
          try {
            await api.meta();
            setLocked(false);
          } catch (err) {
            setLocked(err instanceof ApiError && err.isAuthError);
          }
        }
      } catch {
        // Backend unreachable — show the app and let the views report it,
        // rather than trapping the user behind a gate we cannot verify.
        if (!cancelled) setLocked(false);
      } finally {
        if (!cancelled) setGateChecked(true);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  /* ---------------------------------------------------------------- meta
     Fetched inside the effect with a cancellation flag rather than through a
     callback, so nothing writes state after the component unmounts and the
     effect has no dependency on `disease` (which would otherwise re-fetch
     server metadata every time the user changed condition). `metaNonce` is
     the explicit refresh handle. */
  const [metaNonce, setMetaNonce] = useState(0);
  const refreshMeta = useCallback(() => setMetaNonce((n) => n + 1), []);

  useEffect(() => {
    if (locked || !gateChecked) return undefined;
    let cancelled = false;

    (async () => {
      try {
        const data = await api.meta();
        if (cancelled) return;
        setMeta(data);
        setBootError("");
        setDisease((current) => current || data.supported_diseases?.[0] || "");
      } catch (err) {
        if (cancelled) return;
        if (err instanceof ApiError && err.isAuthError) setLocked(true);
        else setBootError(err.message);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [locked, gateChecked, metaNonce]);

  if (!gateChecked) {
    return (
      <div className="gate">
        <Icon.Spinner size={26} />
      </div>
    );
  }

  if (locked) {
    return (
      <AccessGate
        onUnlock={() => {
          setLocked(false);
          refreshMeta();
        }}
      />
    );
  }

  const diseases = meta?.supported_diseases || [];
  const ageValid = Number(age) >= 1 && Number(age) <= 120;
  const ready = Boolean(disease) && ageValid;
  const Current = { chat: Chat, lookup: Lookup, prescription: Prescription, interactions: InteractionChecker }[view];

  const llm = meta?.llm;
  const faiss = meta?.faiss;
  const ocr = meta?.ocr_runtime;

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">
            <Icon.Pill size={19} weight={2} />
          </div>
          <div>
            <div className="brand-name">MediAssist</div>
            <span className="brand-sub">AI medication assistant</span>
          </div>
        </div>

        <div className="topbar-spacer" />

        {llm && (
          <Pill tone={llm.available ? "ok" : "warn"} dot>
            {llm.available ? "AI online" : "rule-based"}
          </Pill>
        )}

        <button
          className="icon-btn"
          onClick={toggleTheme}
          aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
          title="Toggle theme"
        >
          {theme === "dark" ? <Icon.Sun size={17} /> : <Icon.Moon size={17} />}
        </button>
      </header>

      <div className="shell">
        <aside className="rail">
          <nav className="nav" aria-label="Sections">
            {VIEWS.map((item) => {
              const Glyph = item.icon;
              return (
                <button
                  key={item.id}
                  className={`nav-item ${view === item.id ? "active" : ""}`}
                  onClick={() => setView(item.id)}
                  aria-current={view === item.id ? "page" : undefined}
                >
                  <Glyph size={18} />
                  {item.label}
                  {item.tag && <span className="tag">{item.tag}</span>}
                </button>
              );
            })}
          </nav>

          <Card className="card-pad">
            <div className="label" style={{ marginBottom: 10 }}>
              System
            </div>
            <div className="status-list">
              <Status
                on={llm?.available}
                name="Answer engine"
                value={llm?.model ? String(llm.model).split("/").pop() : "offline"}
              />
              <Status
                on={faiss?.index_ready}
                name="Search"
                value={faiss?.drug_count ? `${faiss.drug_count.toLocaleString()} drugs` : "—"}
              />
              <Status
                on={ocr?.available}
                name="Prescription OCR"
                value={ocr?.engines?.join(", ") || "unavailable"}
              />
              <Status
                on={Boolean(meta?.interactions?.rules)}
                name="Interactions"
                value={meta?.interactions ? `${meta.interactions.rules} rules` : "—"}
              />
            </div>
            {llm?.last_error && (
              <p className="tiny" style={{ color: "var(--warn)", margin: "10px 0 0" }}>
                Last AI error: {llm.last_error}
              </p>
            )}
          </Card>
        </aside>

        <main className="content">
          {bootError && (
            <Alert tone="danger" title="Cannot reach the assistant">
              {bootError}
            </Alert>
          )}

          <section className="hero">
            <h1>Understand what you have been prescribed</h1>
            <p>
              Ask about any medicine, scan a prescription, or check what you take for
              interactions. Every answer is drawn from official drug label data — and none of it
              replaces your doctor or pharmacist.
            </p>
            <div className="hero-row">
              <div className="context-bar" style={{ flex: 1, boxShadow: "none" }}>
                <div className="field">
                  <label className="label" htmlFor="disease">
                    Condition
                  </label>
                  <select
                    id="disease"
                    className="select"
                    value={disease}
                    onChange={(e) => setDisease(e.target.value)}
                  >
                    <option value="">Select…</option>
                    {diseases.map((d) => (
                      <option key={d} value={d}>
                        {d.charAt(0).toUpperCase() + d.slice(1)}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="field narrow">
                  <label className="label" htmlFor="age">
                    Age
                  </label>
                  <input
                    id="age"
                    className="input"
                    type="number"
                    min="1"
                    max="120"
                    value={age}
                    placeholder="e.g. 58"
                    onChange={(e) => setAge(e.target.value)}
                  />
                </div>
                <div className="field" style={{ flex: "0 0 auto" }}>
                  <span className="label">Status</span>
                  {ready ? (
                    <Pill tone="ok" dot>
                      Ready
                    </Pill>
                  ) : (
                    <Pill tone="warn" dot>
                      Needed
                    </Pill>
                  )}
                </div>
              </div>
            </div>
            {!ready && (
              <p className="tiny muted" style={{ marginTop: 10 }}>
                Answers are tailored to your condition and age, so both are required before
                asking anything.
              </p>
            )}
          </section>

          <Current disease={disease} age={age} ready={ready} />

          <div className="safety-note">
            <Icon.Shield size={17} />
            <span>
              <strong>This is not medical advice.</strong> MediAssist is an educational tool. It
              cannot diagnose, cannot tell you whether a medicine is safe for you specifically,
              and must never be used to start, stop or change a prescription. For urgent symptoms
              call your local emergency number.
            </span>
          </div>
        </main>
      </div>

      <footer className="footer">
        MediAssist · educational proof-of-concept · drug data from openFDA, DrugBank and MedDRA
      </footer>
    </div>
  );
}

function Status({ on, name, value }) {
  return (
    <div className="status-row">
      <span className={`status-dot ${on ? "on" : "off"}`} />
      <span className="name">{name}</span>
      <span className="val">{value}</span>
    </div>
  );
}
