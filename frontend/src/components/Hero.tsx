import type { Sample } from "../api/types";
import { apiUrl } from "../api/client";

interface HeroProps {
  samples: readonly Sample[];
}

export default function Hero({ samples }: HeroProps) {
  const images = samples.filter((s) => s.image_url).map((s) => apiUrl(s.image_url as string));
  // Duplicate the strip so the marquee loops seamlessly.
  const strip = [...images, ...images];
  return (
    <header className="hero" id="top">
      <h1 className="hero__title display">
        <span className="hero__word">Read</span>
        <span className="hero__strip" aria-hidden="true">
          {strip.length > 0 && (
            <span className="hero__marquee">
              {strip.map((src, i) => (
                <img key={`${src}-${i}`} src={src} alt="" crossOrigin="anonymous" loading={i < 6 ? "eager" : "lazy"} />
              ))}
            </span>
          )}
        </span>
        <span className="hero__word">
          <span className="hero__tag mono bracket">Leaf + Soil</span>
          Field
        </span>
      </h1>

      <a className="pill-btn hero__cta" href="#field">
        Discover
        <span className="pill-btn__dot" aria-hidden="true" />
      </a>

      <p className="hero__copy lede">
        A leaf shows what is wrong. The soil shows why. Use a photo, a soil test, or both —
        and get one clear answer about your plant's health.
      </p>
    </header>
  );
}
