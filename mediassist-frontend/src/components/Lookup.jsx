import { useState } from "react";
import { api } from "../api.js";
import { Accordion, Alert, Card, Empty, Pill } from "./ui.jsx";
import { Icon } from "./icons.jsx";
import { confidenceTone, titleCase } from "../format.js";

const EXAMPLES = ["metformin", "amlodipine", "warfarin", "salbutamol", "atorvastatin"];

export default function Lookup({ disease, age, ready }) {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function run(term) {
    const drug = (term ?? query).trim();
    if (!drug || !ready) return;
    setQuery(drug);
    setBusy(true);
    setError("");
    try {
      setResult(await api.lookup(drug, disease, age));
    } catch (err) {
      setError(err.message);
      setResult(null);
    } finally {
      setBusy(false);
    }
  }

  const match = result?.best_match;
  const notFound = result && result.status === "no_match";

  return (
    <div className="stack">
      <Card>
        <div className="card-head">
          <div>
            <h2>Medicine lookup</h2>
            <p>Search the knowledge base by generic name, brand name or an ingredient.</p>
          </div>
        </div>
        <div className="card-pad stack">
          <form
            className="row"
            onSubmit={(e) => {
              e.preventDefault();
              run();
            }}
          >
            <input
              className="input"
              style={{ flex: 1, minWidth: 200 }}
              placeholder="e.g. metformin, Panadol, amlodipine 5mg"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              disabled={!ready}
            />
            <button className="btn btn-primary" disabled={busy || !ready || !query.trim()}>
              {busy ? <Icon.Spinner size={16} /> : <Icon.Search size={16} />}
              Search
            </button>
          </form>

          <div className="row">
            <span className="tiny muted">Try:</span>
            {EXAMPLES.map((name) => (
              <button key={name} className="suggestion" onClick={() => run(name)} disabled={!ready}>
                {name}
              </button>
            ))}
          </div>
        </div>
      </Card>

      {error && <Alert tone="danger" title="Lookup failed">{error}</Alert>}

      {/* The identity guard refused to name a drug — show why, and offer the
          alternatives it did find rather than a silent empty state. */}
      {notFound && (
        <Card>
          <div className="card-pad stack">
            <Alert tone="warn" title={`"${result.query}" is not in the knowledge base`}>
              {result.identity_note ||
                "No medicine with that name was found. Similar names are listed below, but they may be different medicines."}
            </Alert>
            {result.suggestions?.length > 0 && (
              <div>
                <div className="label" style={{ marginBottom: 8 }}>
                  Did you mean
                </div>
                <div className="chips">
                  {result.suggestions.map((s) => (
                    <button key={s} className="suggestion" onClick={() => run(s)}>
                      {s}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        </Card>
      )}

      {match && (
        <Card className="fade-in">
          <div className="result-head">
            <div style={{ minWidth: 0 }}>
              <h3>{titleCase(match.generic_name_clean || result.query)}</h3>
              {match.generic_name &&
                match.generic_name.toLowerCase() !== match.generic_name_clean?.toLowerCase() && (
                  <p className="result-sub">On the label as: {match.generic_name}</p>
                )}
            </div>
            <div className="spacer" />
            <Pill tone={confidenceTone(result.confidence)} dot>
              {Math.round((result.confidence || 0) * 100)}% match
            </Pill>
            <Pill>{result.match_type}</Pill>
          </div>

          {result.needs_confirmation && (
            <div className="card-pad" style={{ paddingBottom: 0 }}>
              <Alert tone="warn" title="Confirm this is the right medicine">
                {result.identity_note ||
                  "This was not an exact match. Check the name on your prescription before relying on the information below."}
              </Alert>
            </div>
          )}

          <dl className="kv">
            {match.drug_class && (
              <div>
                <dt>Class</dt>
                <dd>{match.drug_class}</dd>
              </div>
            )}
            {match.route && (
              <div>
                <dt>Route</dt>
                <dd>{match.route}</dd>
              </div>
            )}
            {match.brand_names && (
              <div>
                <dt>Also sold as</dt>
                <dd style={{ fontSize: 13 }}>{match.brand_names}</dd>
              </div>
            )}
          </dl>

          <Accordion title="What it's used for" defaultOpen>
            {match.indications}
          </Accordion>
          <Accordion title="Warnings" meta="read this first">
            {match.warnings}
          </Accordion>
          <Accordion title="How it's taken">{match.dosage_and_administration}</Accordion>
          <Accordion title="Who should not take it">{match.contraindications}</Accordion>
          {match.side_effects_buckets && (
            <Accordion title="Side effects">
              {Object.entries(match.side_effects_buckets)
                .map(([bucket, value]) => `${bucket.replace(/_/g, " ")}: ${value}`)
                .join("\n\n")}
            </Accordion>
          )}

          {result.context_highlights?.highlights && (
            <div className="card-pad" style={{ borderTop: "1px solid var(--line)" }}>
              <div className="label" style={{ marginBottom: 9 }}>
                Notes for {disease}, age {age}
              </div>
              <div className="stack" style={{ gap: 8 }}>
                {Object.values(result.context_highlights.highlights)
                  .flat()
                  .map((note, i) => (
                    <div className="safety-note" key={i}>
                      <Icon.Info size={16} />
                      <span>{note}</span>
                    </div>
                  ))}
              </div>
            </div>
          )}
        </Card>
      )}

      {!result && !busy && !error && (
        <Card>
          <Empty title="Search for a medicine">
            Look up any medicine to see what it treats, its warnings, how it is taken and who
            should avoid it — written in plain language.
          </Empty>
        </Card>
      )}
    </div>
  );
}
