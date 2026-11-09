import { LeafIcon } from "./Icons";

const LINKS = [
  { href: "#field", label: "Samples" },
  { href: "#analyze", label: "Analyze" },
  { href: "#how", label: "How it works" },
];

export default function Toolbar() {
  return (
    <nav className="toolbar" aria-label="Main">
      <div className="toolbar__group">
        <a href="#top" className="toolbar__brand" aria-label="Back to top">
          <span style={{ color: "var(--gold)" }}>
            <LeafIcon />
          </span>
        </a>
        {LINKS.map((link) => (
          <a key={link.href} href={link.href}>
            {link.label}
          </a>
        ))}
      </div>
    </nav>
  );
}
