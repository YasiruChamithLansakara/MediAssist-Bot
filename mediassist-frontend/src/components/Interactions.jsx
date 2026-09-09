import { Pill } from "./ui.jsx";
import { Icon } from "./icons.jsx";
import { titleCase } from "../format.js";

/**
 * Renders an interaction report.
 *
 * Shared by the prescription view (which checks automatically) and the
 * standalone checker, so the framing is identical in both places — including
 * the part that matters most: an empty result is never presented as "safe".
 */
export function InteractionReport({ report, compact = false }) {
  if (!report || !report.checked) return null;

  const findings = report.interactions || [];
  const worst = report.highest_severity;

  return (
    <div className="stack" style={{ gap: 12 }}>
      {!compact && (
        <div className="row">
          <span className="label" style={{ margin: 0 }}>
            Checked together
          </span>
          <div className="spacer" />
          {worst === "major" && <Pill tone="danger" dot>needs attention</Pill>}
          {worst === "moderate" && <Pill tone="warn" dot>worth asking about</Pill>}
          {!worst && <Pill tone="ok" dot>nothing found</Pill>}
        </div>
      )}

      <div className="chips">
        {report.checked.map((drug) => (
          <span className="chip" key={drug} style={{ paddingRight: 12 }}>
            {titleCase(drug)}
          </span>
        ))}
      </div>

      {findings.length === 0 ? (
        <div className="safety-note">
          <Icon.Shield size={17} />
          <span>
            <strong>No interactions found between these medicines.</strong> That means nothing
            matched this checker&rsquo;s list — it does <strong>not</strong> mean the combination is
            safe. Your pharmacist can review everything you take together, including doses.
          </span>
        </div>
      ) : (
        <div>
          {findings.map((finding, index) => (
            <div key={index} className={`ix ix-${finding.severity}`}>
              <div className="ix-top">
                <span className="ix-title">{finding.title}</span>
                <Pill tone={finding.severity === "major" ? "danger" : "warn"}>
                  {finding.severity}
                </Pill>
                {finding.duplicate_therapy && <Pill>same class</Pill>}
                {!finding.high_confidence && <Pill>inferred class</Pill>}
              </div>
              <div className="ix-pair">{finding.drugs.map(titleCase).join("  +  ")}</div>
              <p style={{ marginTop: 8 }}>{finding.mechanism}</p>
              <p className="advice">{finding.advice}</p>
              <p className="source">Source: {finding.source}</p>
            </div>
          ))}
        </div>
      )}

      {report.unclassified_drugs?.length > 0 && (
        <div className="safety-note">
          <Icon.Info size={17} />
          <span>
            Not checked: <strong>{report.unclassified_drugs.map(titleCase).join(", ")}</strong>.
            These are not in the interaction table, so any interaction involving them would have
            been missed.
          </span>
        </div>
      )}

      <p className="tiny muted" style={{ margin: 0 }}>
        {report.disclaimer}
      </p>
    </div>
  );
}
