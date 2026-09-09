/* Small formatting helpers shared across views. */

export function confidenceTone(value) {
  const n = Number(value) || 0;
  if (n >= 0.85) return "ok";
  if (n >= 0.6) return "warn";
  return "danger";
}

export function titleCase(value) {
  return String(value || "")
    .split(" ")
    .filter(Boolean)
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}
