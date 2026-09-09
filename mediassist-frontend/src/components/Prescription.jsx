import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api.js";
import { Alert, Card, Empty, Pill } from "./ui.jsx";
import { Icon } from "./icons.jsx";
import { confidenceTone, titleCase } from "../format.js";
import { InteractionReport } from "./Interactions.jsx";

const MAX_BYTES = 5 * 1024 * 1024;

export default function Prescription({ disease, age, ready }) {
  const [file, setFile] = useState(null);
  const [over, setOver] = useState(false);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const [text, setText] = useState("");
  const [error, setError] = useState("");
  const inputRef = useRef(null);

  // Derived from `file` rather than stored in state: setting state inside an
  // effect causes a second render pass for a value that is a pure function of
  // a prop. The effect below exists only to revoke the URL, which is a real
  // external-resource cleanup and belongs in an effect.
  const preview = useMemo(() => (file ? URL.createObjectURL(file) : ""), [file]);

  useEffect(() => {
    if (!preview) return undefined;
    return () => URL.revokeObjectURL(preview);
  }, [preview]);

  function accept(candidate) {
    setError("");
    if (!candidate) return;
    if (!candidate.type.startsWith("image/")) {
      setError("That file is not an image. Upload a photo or scan of the prescription.");
      return;
    }
    if (candidate.size > MAX_BYTES) {
      setError(`Image must be 5 MB or smaller — this one is ${(candidate.size / 1e6).toFixed(1)} MB.`);
      return;
    }
    setFile(candidate);
    setResult(null);
  }

  async function scan() {
    if (!file || !ready) return;
    setBusy(true);
    setError("");
    try {
      const data = await api.prescriptionImage(file, disease, age);
      setResult(data);
      setText(data?.ocr?.text || "");
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function reanalyse() {
    if (!text.trim() || !ready) return;
    setBusy(true);
    setError("");
    try {
      const data = await api.prescriptionText(text, disease, age);
      setResult((prev) => ({ ...data, ocr: { ...data.ocr, engine: prev?.ocr?.engine } }));
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  function reset() {
    setFile(null);
    setResult(null);
    setText("");
    setError("");
    if (inputRef.current) inputRef.current.value = "";
  }

  const medicines = result?.detected_medicines || [];
  const confidence = result?.ocr?.confidence;

  return (
    <div className="stack">
      <Card>
        <div className="card-head">
          <div>
            <h2>Read a prescription</h2>
            <p>Upload a photo. Text is extracted, medicines identified, and interactions checked.</p>
          </div>
          <div className="spacer" />
          {result && (
            <button className="btn btn-quiet btn-sm" onClick={reset}>
              <Icon.Trash size={15} /> Start over
            </button>
          )}
        </div>

        <div className="card-pad stack">
          {!file && (
            <div
              className={`drop ${over ? "over" : ""}`}
              role="button"
              tabIndex={0}
              onClick={() => inputRef.current?.click()}
              onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && inputRef.current?.click()}
              onDragOver={(e) => {
                e.preventDefault();
                setOver(true);
              }}
              onDragLeave={() => setOver(false)}
              onDrop={(e) => {
                e.preventDefault();
                setOver(false);
                accept(e.dataTransfer.files?.[0]);
              }}
            >
              <Icon.Upload className="drop-icon" />
              <h3>Drop a prescription image here</h3>
              <p>or click to choose a file — JPG or PNG, up to 5 MB</p>
            </div>
          )}

          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            hidden
            onChange={(e) => accept(e.target.files?.[0])}
          />

          {file && (
            <div className="preview">
              {preview && <img src={preview} alt="Prescription preview" />}
              <div className="stack" style={{ gap: 10, flex: 1, minWidth: 0 }}>
                <div>
                  <div style={{ fontWeight: 700 }}>{file.name}</div>
                  <div className="tiny muted">{(file.size / 1024).toFixed(0)} KB</div>
                </div>
                <div className="row">
                  <button className="btn btn-primary" onClick={scan} disabled={busy || !ready}>
                    {busy ? <Icon.Spinner size={16} /> : <Icon.Scan size={16} />}
                    {result ? "Scan again" : "Scan prescription"}
                  </button>
                  <button className="btn btn-ghost btn-sm" onClick={reset} disabled={busy}>
                    Remove
                  </button>
                </div>
                {busy && (
                  <span className="tiny muted">
                    Reading the image — the first scan can take a moment while the OCR model loads.
                  </span>
                )}
              </div>
            </div>
          )}

          {error && <Alert tone="danger" title="Could not process the image">{error}</Alert>}
        </div>
      </Card>

      {result && (
        <>
          <Card className="fade-in">
            <div className="card-head">
              <div>
                <h2>Extracted text</h2>
                <p>OCR is imperfect. Correct anything wrong, then re-analyse.</p>
              </div>
              <div className="spacer" />
              {result.ocr?.engine && <Pill>{result.ocr.engine}</Pill>}
              {typeof confidence === "number" && (
                <Pill tone={confidenceTone(confidence)} dot>
                  {Math.round(confidence * 100)}% confident
                </Pill>
              )}
            </div>
            <div className="card-pad stack">
              {typeof confidence === "number" && confidence < 0.6 && (
                <Alert tone="warn" title="Low OCR confidence">
                  The image was hard to read. Check every medicine name below against the paper
                  before relying on it — a brighter, straighter photo usually helps.
                </Alert>
              )}
              <textarea
                className="textarea"
                value={text}
                onChange={(e) => setText(e.target.value)}
                spellCheck={false}
                aria-label="Extracted prescription text"
              />
              <div className="row">
                <button className="btn btn-ghost" onClick={reanalyse} disabled={busy || !text.trim()}>
                  {busy ? <Icon.Spinner size={16} /> : <Icon.Scan size={16} />}
                  Re-analyse corrected text
                </button>
              </div>
            </div>
          </Card>

          <Card className="fade-in">
            <div className="card-head">
              <div>
                <h2>Medicines found</h2>
                <p>{medicines.length} identified on this prescription.</p>
              </div>
            </div>
            <div className="card-pad">
              {medicines.length === 0 ? (
                <Empty icon={Icon.Pill} title="No medicines recognised">
                  Nothing in the text matched the knowledge base. Correct the text above and
                  re-analyse — OCR errors in a drug name are the usual cause.
                </Empty>
              ) : (
                <div className="med-list">
                  {medicines.map((med, index) => (
                    <div className="med" key={index}>
                      <span className="med-index">{index + 1}</span>
                      <div style={{ minWidth: 0, flex: 1 }}>
                        <div className="row" style={{ gap: 8 }}>
                          <span className="med-name">{titleCase(med.drug || med.text)}</span>
                          {typeof med.confidence === "number" && (
                            <Pill tone={confidenceTone(med.confidence)}>
                              {Math.round(med.confidence * 100)}%
                            </Pill>
                          )}
                        </div>
                        {(med.dosage || med.frequency || med.route) && (
                          <div className="med-facts">
                            {med.dosage && <span className="fact">{med.dosage}</span>}
                            {med.frequency && <span className="fact">{med.frequency}</span>}
                            {med.route && <span className="fact">{med.route}</span>}
                          </div>
                        )}
                        {med.best_match?.indications && (
                          <p className="tiny muted" style={{ margin: "8px 0 0" }}>
                            {String(med.best_match.indications).slice(0, 180)}…
                          </p>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </Card>

          {result.interactions?.checked?.length >= 2 && (
            <Card className="fade-in">
              <div className="card-head">
                <div>
                  <h2>Interactions on this prescription</h2>
                  <p>Every pair of medicines above, checked against each other.</p>
                </div>
              </div>
              <div className="card-pad">
                <InteractionReport report={result.interactions} compact />
              </div>
            </Card>
          )}

          <div className="safety-note">
            <Icon.Shield size={17} />
            <span>
              {result.note ||
                "Educational demo only. OCR can misread a prescription — verify every medicine with a pharmacist."}
            </span>
          </div>
        </>
      )}

      {!file && !result && (
        <Card>
          <Empty icon={Icon.Scan} title="Nothing scanned yet">
            Upload a prescription photo to extract the text, identify each medicine, and check
            them against each other for interactions.
          </Empty>
        </Card>
      )}
    </div>
  );
}
