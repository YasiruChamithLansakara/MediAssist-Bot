/* Shared primitives and icons.
   Icons are inline SVG rather than a library: there are a dozen of them, and
   an icon package would be a larger download than the entire app bundle. */

import { useState } from "react";
import { Icon } from "./icons.jsx";

/* ------------------------------------------------------------ primitives */
export function Pill({ tone = "", children, dot = false }) {
  return (
    <span className={`pill ${tone ? `pill-${tone}` : ""}`}>
      {dot && <i className="dot" />}
      {children}
    </span>
  );
}

export function Card({ children, className = "" }) {
  return <section className={`card ${className}`}>{children}</section>;
}

export function Alert({ tone = "info", title, children, icon = true }) {
  const Glyph = tone === "danger" ? Icon.Alert : tone === "warn" ? Icon.Alert : Icon.Info;
  return (
    <div className={`alert alert-${tone}`} role={tone === "danger" ? "alert" : "status"}>
      {icon && <Glyph size={19} />}
      <div>
        {title && <h3>{title}</h3>}
        <p>{children}</p>
      </div>
    </div>
  );
}

export function Empty({ icon: Glyph = Icon.Search, title, children }) {
  return (
    <div className="empty">
      <Glyph size={34} weight={1.4} />
      <h3>{title}</h3>
      <p>{children}</p>
    </div>
  );
}

export function Accordion({ title, children, defaultOpen = false, meta }) {
  const [open, setOpen] = useState(defaultOpen);
  if (!children) return null;
  return (
    <div className="accordion">
      <button className="acc-btn" aria-expanded={open} onClick={() => setOpen((v) => !v)}>
        <span>{title}</span>
        {meta && <span className="tiny muted mono">{meta}</span>}
        <Icon.Chevron className="caret" />
      </button>
      {open && <div className="acc-panel fade-in">{children}</div>}
    </div>
  );
}

/* --------------------------------------------------------------- markdown
   The LLM replies in a small, predictable subset of Markdown — bold, bullet
   lists, headings and horizontal rules. A full parser would be a much larger
   dependency than the six rules actually needed. */
function inline(text) {
  const nodes = [];
  const pattern = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g;
  let last = 0;
  let match;
  while ((match = pattern.exec(text)) !== null) {
    if (match.index > last) nodes.push(text.slice(last, match.index));
    const token = match[0];
    const key = `${match.index}-${token.length}`;
    if (token.startsWith("**")) nodes.push(<strong key={key}>{token.slice(2, -2)}</strong>);
    else if (token.startsWith("`")) nodes.push(<code key={key}>{token.slice(1, -1)}</code>);
    else nodes.push(<em key={key}>{token.slice(1, -1)}</em>);
    last = match.index + token.length;
  }
  if (last < text.length) nodes.push(text.slice(last));
  return nodes;
}

export function Markdown({ text }) {
  if (!text) return null;
  const blocks = [];
  let list = [];

  const flush = () => {
    if (list.length) {
      blocks.push(
        <ul key={`ul-${blocks.length}`}>
          {list.map((item, i) => (
            <li key={i}>{inline(item)}</li>
          ))}
        </ul>,
      );
      list = [];
    }
  };

  String(text)
    .split("\n")
    .forEach((raw, index) => {
      const line = raw.trimEnd();
      if (/^\s*[-•*]\s+/.test(line)) {
        list.push(line.replace(/^\s*[-•*]\s+/, ""));
        return;
      }
      flush();
      if (!line.trim()) return;
      if (/^-{3,}$/.test(line.trim())) {
        blocks.push(<hr key={`hr-${index}`} />);
      } else if (/^\*\*[^*]+\*\*:?\s*$/.test(line.trim())) {
        blocks.push(<h4 key={`h-${index}`}>{line.replace(/\*\*/g, "").replace(/:$/, "")}</h4>);
      } else {
        blocks.push(<p key={`p-${index}`}>{inline(line)}</p>);
      }
    });

  flush();
  return <>{blocks}</>;
}
