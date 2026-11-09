/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Base URL of the FastAPI backend, e.g. https://plant-api.vercel.app (empty = same origin). */
  readonly VITE_API_URL?: string;
  /** Link to the source repository shown in the footer. */
  readonly VITE_REPO_URL?: string;
}
