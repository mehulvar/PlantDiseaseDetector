const REPO_URL = import.meta.env.VITE_REPO_URL;

export default function Footer() {
  return (
    <footer className="footer mono">
      <span className="bracket">Powered by ONNX + FastAPI</span>
      <span>
        Decision support only — confirm serious problems with a local agronomist.
        {REPO_URL && (
          <>
            {" "}
            <a href={REPO_URL} target="_blank" rel="noreferrer">
              Source code
            </a>
          </>
        )}
      </span>
    </footer>
  );
}
