import { memo } from "react";

/** A stylised aerial patchwork of crop plots, drawn in a 1600x1000 coordinate space. */

export const FIELD_WIDTH = 1600;
export const FIELD_HEIGHT = 1000;

type Point = readonly [number, number];

// Jittered grid of plot corners (5 columns x 4 rows of nodes -> 4 x 3 plots).
const NODES: readonly (readonly Point[])[] = [
  [[0, 0], [380, 0], [790, 0], [1200, 0], [1600, 0]],
  [[0, 330], [395, 305], [770, 350], [1215, 318], [1600, 340]],
  [[0, 680], [370, 702], [805, 662], [1190, 690], [1600, 668]],
  [[0, 1000], [385, 1000], [780, 1000], [1210, 1000], [1600, 1000]],
];

const PLOT_COLORS = [
  "#7f9a5a", "#a9b878", "#c8b571", "#6c8a4c",
  "#b39069", "#8faa66", "#9d8fb8", "#7a9655",
  "#c2a878", "#5f7f45", "#a4b36f", "#8b9f5e",
];
const ROW_ANGLES = [0, 32, -24, 68];
const POND_PLOT = 5;
const PLOT_INSET = 0.92;

interface Plot {
  index: number;
  points: readonly Point[];
  centroid: Point;
}

function centroidOf(points: readonly Point[]): Point {
  const x = points.reduce((sum, p) => sum + p[0], 0) / points.length;
  const y = points.reduce((sum, p) => sum + p[1], 0) / points.length;
  return [x, y];
}

function inset(points: readonly Point[], factor: number): Point[] {
  const [cx, cy] = centroidOf(points);
  return points.map(([x, y]) => [cx + (x - cx) * factor, cy + (y - cy) * factor]);
}

const PLOTS: readonly Plot[] = NODES.slice(0, -1).flatMap((row, r) =>
  row.slice(0, -1).map((_, c) => {
    const corners = [NODES[r][c], NODES[r][c + 1], NODES[r + 1][c + 1], NODES[r + 1][c]];
    return { index: r * 4 + c, points: inset(corners, PLOT_INSET), centroid: centroidOf(corners) };
  }),
);

/** Pin anchor points (percent of the field) - one per crop plot, skipping the farmyard. */
export const PIN_POSITIONS: readonly { x: number; y: number }[] = PLOTS.filter(
  (p) => p.index !== POND_PLOT,
).map((p) => ({ x: (p.centroid[0] / FIELD_WIDTH) * 100, y: (p.centroid[1] / FIELD_HEIGHT) * 100 }));

// Deterministic pseudo-random numbers so the trees never jump between renders.
function seeded(seed: number): () => number {
  let state = seed;
  return () => {
    state = (state * 16807) % 2147483647;
    return state / 2147483647;
  };
}

const TREES = (() => {
  const random = seeded(7);
  const hedge = Array.from({ length: 34 }, (_, i) => ({
    x: 20 + i * 47 + random() * 10,
    y: 686 + Math.sin(i / 2.4) * 12 + random() * 8,
    r: 13 + random() * 9,
  }));
  const grove = Array.from({ length: 16 }, () => ({
    x: 1420 + random() * 160,
    y: 30 + random() * 260,
    r: 16 + random() * 12,
  }));
  return [...hedge, ...grove];
})();

const toPath = (points: readonly Point[]) => points.map(([x, y]) => `${x},${y}`).join(" ");

function FieldIllustration() {
  const pond = PLOTS[POND_PLOT];
  const [px, py] = pond.centroid;
  return (
    <svg viewBox={`0 0 ${FIELD_WIDTH} ${FIELD_HEIGHT}`} aria-hidden="true" preserveAspectRatio="none">
      <defs>
        {ROW_ANGLES.map((angle, i) => (
          <pattern
            key={angle}
            id={`rows-${i}`}
            width="16"
            height="16"
            patternUnits="userSpaceOnUse"
            patternTransform={`rotate(${angle})`}
          >
            <rect width="16" height="6" fill="rgba(25, 35, 15, 0.2)" />
          </pattern>
        ))}
        <radialGradient id="pond" cx="50%" cy="45%" r="60%">
          <stop offset="0%" stopColor="#8fb8c6" />
          <stop offset="100%" stopColor="#4f7f93" />
        </radialGradient>
        <linearGradient id="sun" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="rgba(255, 244, 214, 0.35)" />
          <stop offset="60%" stopColor="rgba(255, 244, 214, 0)" />
        </linearGradient>
      </defs>

      <rect width={FIELD_WIDTH} height={FIELD_HEIGHT} fill="#d8ccb2" />

      {PLOTS.filter((p) => p.index !== POND_PLOT).map((plot) => (
        <g key={plot.index}>
          <polygon points={toPath(plot.points)} fill={PLOT_COLORS[plot.index]} />
          <polygon points={toPath(plot.points)} fill={`url(#rows-${plot.index % ROW_ANGLES.length})`} />
        </g>
      ))}

      {/* Farmyard: grass, pond and barn */}
      <polygon points={toPath(pond.points)} fill="#9fb27a" />
      <ellipse cx={px - 70} cy={py + 20} rx="120" ry="78" fill="url(#pond)" />
      <ellipse cx={px - 70} cy={py + 20} rx="120" ry="78" fill="none" stroke="#e8e0cc" strokeWidth="6" />
      <g transform={`translate(${px + 70} ${py - 70})`}>
        <rect width="120" height="80" rx="6" fill="#9b3d2b" />
        <rect y="-14" width="120" height="22" rx="6" fill="#5b2a20" />
        <rect x="46" y="38" width="28" height="42" fill="#e8e0cc" />
      </g>
      <path
        d={`M ${px + 130} ${py + 10} C ${px + 160} ${py + 120}, ${px + 40} ${py + 150}, ${px + 10} 680`}
        stroke="#e8e0cc"
        strokeWidth="14"
        fill="none"
        strokeLinecap="round"
      />

      {TREES.map((t, i) => (
        <g key={i}>
          <circle cx={t.x + 4} cy={t.y + 6} r={t.r} fill="rgba(20, 30, 10, 0.25)" />
          <circle cx={t.x} cy={t.y} r={t.r} fill={i % 3 === 0 ? "#3f5f34" : "#4d7040"} />
        </g>
      ))}

      <rect width={FIELD_WIDTH} height={FIELD_HEIGHT} fill="url(#sun)" />
    </svg>
  );
}

export default memo(FieldIllustration);
