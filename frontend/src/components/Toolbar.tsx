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
        <a href="#top" className="toolbar__brand" aria-label="AgroAI, back to top">
          <span style={{ color: "var(--gold)" }}>
            <LeafIcon />
          </span>
          <span className="toolbar__name">AgroAI</span>
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
