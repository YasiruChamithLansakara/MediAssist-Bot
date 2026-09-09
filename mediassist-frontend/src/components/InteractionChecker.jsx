import { useState } from "react";
import { api } from "../api.js";
import { Alert, Card, Empty } from "./ui.jsx";
import { Icon } from "./icons.jsx";
import { titleCase } from "../format.js";
import { InteractionReport } from "./Interactions.jsx";

const PRESETS = [
  { label: "Warfarin + ibuprofen", drugs: ["warfarin", "ibuprofen"] },
  { label: "Lisinopril + spironolactone", drugs: ["lisinopril", "spironolactone"] },
  { label: "Propranolol + salbutamol", drugs: ["propranolol", "salbutamol"] },
];

export default function InteractionChecker({ disease, ready }) {
  const [drugs, setDrugs] = useState([]);
  const [entry, setEntry] = useState("");
  const [report, setReport] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  function add(name) {
    const value = (name ?? entry).trim();
    if (!value) return;
    const key = value.toLowerCase();
    if (!drugs.some((d) => d.toLowerCase() === key)) {
      setDrugs((prev) => [...prev, value]);
      setReport(null);
    }
    setEntry("");
  }

  function remove(name) {
    setDrugs((prev) => prev.filter((d) => d !== name));
    setReport(null);
  }

  async function check(list) {
    const target = list ?? drugs;
    if (target.length < 2) return;
    setBusy(true);
    setError("");
    try {
      setReport(await api.interactions(target, disease));
    } catch (err) {
      setError(err.message);
      setReport(null);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="stack">
      <Card>
        <div className="card-head">
          <div>
            <h2>Interaction checker</h2>
            <p>Add every medicine you take — including ones bought without a prescription.</p>
          </div>
        </div>

        <div className="card-pad stack">
          <form
            className="row"
            onSubmit={(e) => {
              e.preventDefault();
              add();
            }}
          >
            <input
              className="input"
              style={{ flex: 1, minWidth: 190 }}
              placeholder="Add a medicine, e.g. warfarin"
              value={entry}
              onChange={(e) => setEntry(e.target.value)}
              disabled={!ready}
            />
            <button className="btn btn-ghost" disabled={!entry.trim()}>
              Add
            </button>
          </form>

          {drugs.length > 0 && (
            <div className="chips">
              {drugs.map((drug) => (
                <span className="chip" key={drug}>
                  {titleCase(drug)}
                  <button onClick={() => remove(drug)} aria-label={`Remove ${drug}`}>
                    <Icon.Close />
                  </button>
                </span>
              ))}
            </div>
          )}

          <div className="row">
            <button
              className="btn btn-primary"
              disabled={busy || drugs.length < 2 || !ready}
              onClick={() => check()}
            >
              {busy ? <Icon.Spinner size={16} /> : <Icon.Link size={16} />}
              Check {drugs.length >= 2 ? `${drugs.length} medicines` : "interactions"}
            </button>
            {drugs.length === 1 && <span className="tiny muted">Add one more to check.</span>}
          </div>

          {drugs.length === 0 && (
            <div className="row">
              <span className="tiny muted">Examples:</span>
              {PRESETS.map((preset) => (
                <button
                  key={preset.label}
                  className="suggestion"
                  disabled={!ready}
                  onClick={() => {
                    setDrugs(preset.drugs);
                    check(preset.drugs);
                  }}
                >
                  {preset.label}
                </button>
              ))}
            </div>
          )}
        </div>
      </Card>

      {error && (
        <Alert tone="danger" title="Check failed">
          {error}
        </Alert>
      )}

      {report && (
        <Card className="fade-in">
          <div className="card-pad">
            <InteractionReport report={report} />
          </div>
        </Card>
      )}

      {!report && !busy && !error && (
        <Card>
          <Empty icon={Icon.Link} title="Check medicines against each other">
            A chronic condition often means several medicines from several doctors. This checks
            documented interactions between them — mechanism, what to watch for, and the source.
          </Empty>
        </Card>
      )}
    </div>
  );
}
