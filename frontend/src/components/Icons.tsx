const common = { fill: "none", stroke: "currentColor", strokeWidth: 2, strokeLinecap: "round", strokeLinejoin: "round" } as const;

export const LeafIcon = () => (
  <svg viewBox="0 0 24 24" aria-hidden="true" {...common}>
    <path d="M5 19c0-8 5-13 14-14-1 9-6 14-14 14z" fill="currentColor" stroke="none" />
    <path d="M5 19l8-8" stroke="var(--ink)" />
  </svg>
);

export const ChevronLeft = () => (
  <svg viewBox="0 0 24 24" aria-hidden="true" {...common}>
    <path d="M15 5l-7 7 7 7" />
  </svg>
);

export const ChevronRight = () => (
  <svg viewBox="0 0 24 24" aria-hidden="true" {...common}>
    <path d="M9 5l7 7-7 7" />
  </svg>
);

export const CloseIcon = () => (
  <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true" {...common}>
    <path d="M6 6l12 12M18 6L6 18" />
  </svg>
);

export const UploadIcon = () => (
  <svg viewBox="0 0 56 56" className="dropzone__icon" aria-hidden="true" {...common} strokeWidth={1.6}>
    <rect x="4" y="4" width="48" height="48" rx="24" />
    <path d="M28 36V18M20 25l8-8 8 8" />
  </svg>
);
