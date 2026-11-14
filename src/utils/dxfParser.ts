import type { ParsedDxfSummary } from '../types/analysis';

export interface DxfLinePrimitive {
  kind: 'line';
  layer: string;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface DxfCirclePrimitive {
  kind: 'circle';
  layer: string;
  cx: number;
  cy: number;
  r: number;
}

export interface DxfArcPrimitive {
  kind: 'arc';
  layer: string;
  cx: number;
  cy: number;
  r: number;
  startDeg: number;
  endDeg: number;
}

export interface DxfTextPrimitive {
  kind: 'text';
  layer: string;
  x: number;
  y: number;
  text: string;
  height: number;
}

export type DxfPrimitive =
  | DxfLinePrimitive
  | DxfCirclePrimitive
  | DxfArcPrimitive
  | DxfTextPrimitive;

export interface ParsedDxfResult {
  summary: ParsedDxfSummary;
  primitives: DxfPrimitive[];
  svgMarkup: string;
}

const INSUNITS_MAP: Record<number, string> = {
  0: 'Unitless (assume metres for architectural CAD)',
  1: 'Inches',
  2: 'Feet',
  4: 'Millimetres',
  5: 'Centimetres',
  6: 'Metres',
};

function cleanMText(raw: string): string {
  return raw
    .replace(/\\P/gi, ' ')
    .replace(/\{\\[A-Za-z][^;]*;/g, '')
    .replace(/[{}]/g, '')
    .replace(/\\[A-Za-z][^;]*;/g, '')
    .trim();
}

export function parseDxfFile(text: string): ParsedDxfResult {
  const lines = text.replace(/\r\n/g, '\n').split('\n');
  const pairs: Array<{ code: number; val: string }> = [];
  for (let i = 0; i + 1 < lines.length; i += 2) {
    const code = parseInt(lines[i].trim(), 10);
    if (!Number.isNaN(code)) {
      pairs.push({ code, val: lines[i + 1].trim() });
    }
  }

  let units = 'Metres';
  let inHeader = false;
  let currentVar = '';
  let inEntities = false;

  const layerCounts = new Map<string, number>();
  const entityCounts: Record<string, number> = {};
  const blockCounts = new Map<string, number>();
  const texts: string[] = [];
  const primitives: DxfPrimitive[] = [];

  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  const expandBounds = (x: number, y: number) => {
    if (!Number.isFinite(x) || !Number.isFinite(y)) return;
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
  };

  let idx = 0;
  while (idx < pairs.length) {
    const { code, val } = pairs[idx];
    if (code === 0 && val === 'SECTION') {
      const next = pairs[idx + 1];
      if (next?.code === 2) {
        inHeader = next.val === 'HEADER';
        inEntities = next.val === 'ENTITIES';
        idx += 2;
        continue;
      }
    }
    if (code === 0 && val === 'ENDSEC') {
      inHeader = false;
      inEntities = false;
      idx++;
      continue;
    }

    if (inHeader) {
      if (code === 9) {
        currentVar = val;
      } else if (currentVar === '$INSUNITS' && code === 70) {
        units = INSUNITS_MAP[parseInt(val, 10)] ?? units;
      }
      idx++;
      continue;
    }

    if (inEntities && code === 0) {
      const entityType = val;
      idx++;
      const attrs: Array<{ code: number; val: string }> = [];
      while (idx < pairs.length && pairs[idx].code !== 0) {
        attrs.push(pairs[idx]);
        idx++;
      }

      if (entityType === 'ENDSEC' || entityType === 'SEQEND' || entityType === 'VERTEX') {
        continue;
      }

      entityCounts[entityType] = (entityCounts[entityType] ?? 0) + 1;

      let layer = '0';
      for (const a of attrs) {
        if (a.code === 8 && a.val) {
          layer = a.val;
          break;
        }
      }
      layerCounts.set(layer, (layerCounts.get(layer) ?? 0) + 1);

      const getFirstNum = (c: number, fb = 0) => {
        const hit = attrs.find((a) => a.code === c);
        const n = hit ? parseFloat(hit.val) : fb;
        return Number.isFinite(n) ? n : fb;
      };
      const getFirstStr = (c: number) => attrs.find((a) => a.code === c)?.val ?? '';

      if (entityType === 'LINE') {
        const x1 = getFirstNum(10);
        const y1 = getFirstNum(20);
        const x2 = getFirstNum(11);
        const y2 = getFirstNum(21);
        expandBounds(x1, y1);
        expandBounds(x2, y2);
        primitives.push({ kind: 'line', layer, x1, y1, x2, y2 });
      } else if (entityType === 'LWPOLYLINE') {
        const xs = attrs.filter((a) => a.code === 10).map((a) => parseFloat(a.val));
        const ys = attrs.filter((a) => a.code === 20).map((a) => parseFloat(a.val));
        const closed = (getFirstNum(70, 0) & 1) === 1;
        const len = Math.min(xs.length, ys.length);
        for (let i = 0; i < len; i++) {
          expandBounds(xs[i], ys[i]);
          if (i > 0) {
            primitives.push({
              kind: 'line',
              layer,
              x1: xs[i - 1],
              y1: ys[i - 1],
              x2: xs[i],
              y2: ys[i],
            });
          }
        }
        if (closed && len > 2) {
          primitives.push({
            kind: 'line',
            layer,
            x1: xs[len - 1],
            y1: ys[len - 1],
            x2: xs[0],
            y2: ys[0],
          });
        }
      } else if (entityType === 'CIRCLE') {
        const cx = getFirstNum(10);
        const cy = getFirstNum(20);
        const r = getFirstNum(40);
        expandBounds(cx - r, cy - r);
        expandBounds(cx + r, cy + r);
        primitives.push({ kind: 'circle', layer, cx, cy, r });
      } else if (entityType === 'ARC') {
        const cx = getFirstNum(10);
        const cy = getFirstNum(20);
        const r = getFirstNum(40);
        const startDeg = getFirstNum(50);
        const endDeg = getFirstNum(51);
        expandBounds(cx - r, cy - r);
        expandBounds(cx + r, cy + r);
        primitives.push({ kind: 'arc', layer, cx, cy, r, startDeg, endDeg });
      } else if (entityType === 'TEXT' || entityType === 'MTEXT') {
        const x = getFirstNum(10);
        const y = getFirstNum(20);
        const h = getFirstNum(40, 0.25);
        const raw =
          entityType === 'MTEXT'
            ? attrs
                .filter((a) => a.code === 3 || a.code === 1)
                .map((a) => a.val)
                .join('')
            : getFirstStr(1);
        const cleaned = cleanMText(raw);
        if (cleaned) {
          texts.push(cleaned);
          expandBounds(x, y);
          primitives.push({ kind: 'text', layer, x, y, text: cleaned, height: h });
        }
      } else if (entityType === 'INSERT') {
        const blockName = getFirstStr(2);
        if (blockName) {
          blockCounts.set(blockName, (blockCounts.get(blockName) ?? 0) + 1);
        }
        const x = getFirstNum(10);
        const y = getFirstNum(20);
        expandBounds(x, y);
        primitives.push({ kind: 'circle', layer, cx: x, cy: y, r: 0.15 });
      }
      continue;
    }

    idx++;
  }

  const hasBounds = Number.isFinite(minX) && Number.isFinite(maxX) && maxX > minX && maxY > minY;
  const width = hasBounds ? maxX - minX : 10;
  const height = hasBounds ? maxY - minY : 10;
  const pad = Math.max(width, height) * 0.06;

  // Render crisp SVG representation (flipped Y because CAD Y is up while SVG Y is down)
  const vbX = (hasBounds ? minX : 0) - pad;
  const vbY = -(hasBounds ? maxY : 10) - pad;
  const vbW = width + pad * 2;
  const vbH = height + pad * 2;
  const strokeW = Math.max(vbW, vbH) * 0.0022;

  const layerColor = (layer: string) => {
    const l = layer.toLowerCase();
    if (l.includes('wall') || l.includes('wand') || l.includes('struct')) return '#0f172a';
    if (l.includes('door') || l.includes('tuer')) return '#2563eb';
    if (l.includes('win') || l.includes('fenst')) return '#0284c7';
    if (l.includes('light') || l.includes('lamp') || l.includes('elec')) return '#d97706';
    if (l.includes('furn') || l.includes('fix')) return '#64748b';
    if (l.includes('text') || l.includes('room') || l.includes('label')) return '#1e293b';
    return '#334155';
  };

  const svgParts: string[] = [];
  // Cap primitives rendered into SVG DOM to keep UI snappy
  const maxPrims = Math.min(primitives.length, 6000);
  for (let i = 0; i < maxPrims; i++) {
    const p = primitives[i];
    const col = layerColor(p.layer);
    if (p.kind === 'line') {
      svgParts.push(
        `<line x1="${p.x1.toFixed(3)}" y1="${(-p.y1).toFixed(3)}" x2="${p.x2.toFixed(3)}" y2="${(-p.y2).toFixed(3)}" stroke="${col}" stroke-width="${strokeW.toFixed(4)}" stroke-linecap="round"/>`,
      );
    } else if (p.kind === 'circle') {
      svgParts.push(
        `<circle cx="${p.cx.toFixed(3)}" cy="${(-p.cy).toFixed(3)}" r="${p.r.toFixed(3)}" fill="none" stroke="${col}" stroke-width="${strokeW.toFixed(4)}"/>`,
      );
    } else if (p.kind === 'arc') {
      const a1 = (p.startDeg * Math.PI) / 180;
      const a2 = (p.endDeg * Math.PI) / 180;
      const x1 = p.cx + p.r * Math.cos(a1);
      const y1 = -(p.cy + p.r * Math.sin(a1));
      const x2 = p.cx + p.r * Math.cos(a2);
      const y2 = -(p.cy + p.r * Math.sin(a2));
      let sweep = (p.endDeg - p.startDeg + 360) % 360;
      const largeArc = sweep > 180 ? 1 : 0;
      svgParts.push(
        `<path d="M ${x1.toFixed(3)} ${y1.toFixed(3)} A ${p.r.toFixed(3)} ${p.r.toFixed(3)} 0 ${largeArc} 0 ${x2.toFixed(3)} ${y2.toFixed(3)}" fill="none" stroke="${col}" stroke-width="${strokeW.toFixed(4)}"/>`,
      );
    } else if (p.kind === 'text') {
      const fontSize = Math.max(p.height * 0.85, strokeW * 5);
      const escaped = p.text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
      svgParts.push(
        `<text x="${p.x.toFixed(3)}" y="${(-p.y).toFixed(3)}" fill="#0f172a" font-family="DM Sans, sans-serif" font-size="${fontSize.toFixed(4)}">${escaped}</text>`,
      );
    }
  }

  const svgMarkup = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="${vbX.toFixed(3)} ${vbY.toFixed(3)} ${vbW.toFixed(3)} ${vbH.toFixed(3)}" style="background:#ffffff">${svgParts.join('')}</svg>`;

  return {
    summary: {
      units,
      layers: Array.from(layerCounts.entries())
        .map(([name, entityCount]) => ({ name, entityCount }))
        .sort((a, b) => b.entityCount - a.entityCount),
      entityCounts,
      blocks: Array.from(blockCounts.entries())
        .map(([name, count]) => ({ name, count }))
        .sort((a, b) => b.count - a.count),
      texts: Array.from(new Set(texts)).slice(0, 120),
      boundingBox: hasBounds
        ? {
            minX: Number(minX.toFixed(2)),
            minY: Number(minY.toFixed(2)),
            maxX: Number(maxX.toFixed(2)),
            maxY: Number(maxY.toFixed(2)),
            width: Number(width.toFixed(2)),
            height: Number(height.toFixed(2)),
          }
        : null,
    },
    primitives,
    svgMarkup,
  };
}
