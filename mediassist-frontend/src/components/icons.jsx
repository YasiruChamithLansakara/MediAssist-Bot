/* Inline SVG icon set.
   Kept in its own module so `ui.jsx` exports only components — React Fast
   Refresh silently stops working for any file that mixes component and
   non-component exports. */

const ico = (props) => ({
  width: props.size || 18,
  height: props.size || 18,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: props.weight || 1.8,
  strokeLinecap: "round",
  strokeLinejoin: "round",
  "aria-hidden": true,
});

export const Icon = {
  Pill: (p) => (
    <svg {...ico(p)}>
      <path d="M10.5 20.5a6 6 0 0 1-8.5-8.5l7-7a6 6 0 0 1 8.5 8.5z" />
      <path d="M8.5 8.5l7 7" />
    </svg>
  ),
  Chat: (p) => (
    <svg {...ico(p)}>
      <path d="M21 11.5a8.4 8.4 0 0 1-9 8.4 8.9 8.9 0 0 1-4-.9L3 21l1.9-4.5A8.4 8.4 0 0 1 12 3a8.4 8.4 0 0 1 9 8.5z" />
    </svg>
  ),
  Scan: (p) => (
    <svg {...ico(p)}>
      <path d="M3 8V5a2 2 0 0 1 2-2h3M16 3h3a2 2 0 0 1 2 2v3M21 16v3a2 2 0 0 1-2 2h-3M8 21H5a2 2 0 0 1-2-2v-3" />
      <path d="M7 12h10" />
    </svg>
  ),
  Link: (p) => (
    <svg {...ico(p)}>
      <path d="M10 13a5 5 0 0 0 7.5.5l3-3a5 5 0 0 0-7-7l-1.7 1.7" />
      <path d="M14 11a5 5 0 0 0-7.5-.5l-3 3a5 5 0 0 0 7 7l1.7-1.7" />
    </svg>
  ),
  Alert: (p) => (
    <svg {...ico(p)}>
      <path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z" />
      <path d="M12 9v4M12 17h.01" />
    </svg>
  ),
  Shield: (p) => (
    <svg {...ico(p)}>
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
      <path d="m9 12 2 2 4-4" />
    </svg>
  ),
  Search: (p) => (
    <svg {...ico(p)}>
      <circle cx="11" cy="11" r="7" />
      <path d="m20 20-3.5-3.5" />
    </svg>
  ),
  Send: (p) => (
    <svg {...ico(p)}>
      <path d="M22 2 11 13" />
      <path d="M22 2l-7 20-4-9-9-4z" />
    </svg>
  ),
  Upload: (p) => (
    <svg {...ico({ ...p, size: p.size || 30 })}>
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <path d="M7 9l5-5 5 5M12 4v12" />
    </svg>
  ),
  Chevron: (p) => (
    <svg {...ico({ ...p, size: p.size || 16 })}>
      <path d="m9 18 6-6-6-6" />
    </svg>
  ),
  Sun: (p) => (
    <svg {...ico(p)}>
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
    </svg>
  ),
  Moon: (p) => (
    <svg {...ico(p)}>
      <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" />
    </svg>
  ),
  Spinner: (p) => (
    <svg {...ico(p)} className="spin">
      <path d="M12 3a9 9 0 1 0 9 9" />
    </svg>
  ),
  Close: (p) => (
    <svg {...ico({ ...p, size: p.size || 14 })}>
      <path d="M18 6 6 18M6 6l12 12" />
    </svg>
  ),
  Info: (p) => (
    <svg {...ico(p)}>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 16v-5M12 8h.01" />
    </svg>
  ),
  Trash: (p) => (
    <svg {...ico({ ...p, size: p.size || 16 })}>
      <path d="M3 6h18M8 6V4a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v2M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
    </svg>
  ),
};

