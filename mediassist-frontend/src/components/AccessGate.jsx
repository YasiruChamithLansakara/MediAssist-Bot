import { useState } from "react";
import { Icon } from "./icons.jsx";
import { verifyAccessCode } from "../api.js";

/**
 * The access gate.
 *
 * Deliberately says what this tool is and is not before anyone gets in — a
 * medication assistant that a stranger opens should establish "educational,
 * not medical advice" on the very first screen, not three clicks later.
 */
export default function AccessGate({ onUnlock }) {
  const [code, setCode] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function submit(event) {
    event.preventDefault();
    const trimmed = code.trim();
    if (!trimmed) return;

    setBusy(true);
    setError("");
    try {
      const ok = await verifyAccessCode(trimmed);
      if (ok) onUnlock();
      else setError("That code was not recognised. Check it and try again.");
    } catch (err) {
      setError(err.message || "Could not reach the server.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="gate">
      <form className="gate-card fade-in" onSubmit={submit}>
        <div className="brand-mark">
          <Icon.Pill size={26} weight={2} />
        </div>
        <h1>MediAssist</h1>
        <p>
          An AI assistant that explains prescription medicines in plain language.
          Enter your access code to continue.
        </p>

        <div className="field">
          <label className="label" htmlFor="access-code">
            Access code
          </label>
          <input
            id="access-code"
            className="input"
            type="password"
            value={code}
            /* The gate is a single-purpose screen with exactly one field and
               nothing else to read past. Focusing it is what a keyboard or
               screen-reader user wants here; the rule guards against
               autofocus buried inside a longer page. */
            // eslint-disable-next-line jsx-a11y/no-autofocus
            autoFocus
            autoComplete="off"
            placeholder="••••••••"
            onChange={(e) => setCode(e.target.value)}
            aria-describedby={error ? "gate-error" : undefined}
          />
        </div>

        {error && (
          <p className="gate-err" id="gate-error" role="alert">
            {error}
          </p>
        )}

        <button
          className="btn btn-primary"
          style={{ width: "100%", marginTop: 16 }}
          disabled={busy || !code.trim()}
          type="submit"
        >
          {busy ? <Icon.Spinner size={16} /> : <Icon.Shield size={16} />}
          {busy ? "Checking…" : "Unlock"}
        </button>

        <p className="gate-foot">
          Educational information only — not medical advice, and never a
          substitute for your doctor or pharmacist. In an emergency, call your
          local emergency number.
        </p>
      </form>
    </div>
  );
}
