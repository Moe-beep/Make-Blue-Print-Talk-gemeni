import type {
  ParsedDxfApartment,
  ParsedDxfRoom,
  ParsedDxfSummary,
} from '../types/analysis';

export interface DxfLinePrimitive {
  kind: 'line';
  layer: string;
  blockName?: string;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface DxfCirclePrimitive {
  kind: 'circle';
  layer: string;
  blockName?: string;
  cx: number;
  cy: number;
  r: number;
}

export interface DxfArcPrimitive {
  kind: 'arc';
  layer: string;
  blockName?: string;
  cx: number;
  cy: number;
  r: number;
  startDeg: number;
  endDeg: number;
}

export interface DxfTextPrimitive {
  kind: 'text';
  layer: string;
  blockName?: string;
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

export interface DxfRoomPolygon {
  code: string;
  label: string;
  apartment: string | null;
  areaSqM: number;
  isExterior: boolean;
  pts: Array<{ x: number; y: number }>;
}

export interface ParsedDxfResult {
  summary: ParsedDxfSummary;
  primitives: DxfPrimitive[];
  roomPolygons: DxfRoomPolygon[];
  viewBox: { x: number; y: number; width: number; height: number };
  svgMarkup: string;
}

interface ClosedPoly {
  layer: string;
  pts: Array<{ x: number; y: number }>;
  areaRaw: number;
}

const INSUNITS_MAP: Record<number, { label: string; toMeters: number }> = {
  0: { label: 'Unitless (assumed metres)', toMeters: 1 },
  1: { label: 'Inches', toMeters: 0.0254 },
  2: { label: 'Feet', toMeters: 0.3048 },
  4: { label: 'Millimetres', toMeters: 0.001 },
  5: { label: 'Centimetres', toMeters: 0.01 },
  6: { label: 'Metres', toMeters: 1 },
};

const EXTERIOR_ROOM_REGEX =
  /balkon|balcon|balcone|loggia|terrasse|terrace|patio|porch|aussen|outdoor|shaft|void|schacht|luftraum/i;

function cleanMText(raw: string): string {
  return raw
    .replace(/\\P/gi, ' ')
    .replace(/\{\\[A-Za-z][^;]*;/g, '')
    .replace(/[{}]/g, '')
    .replace(/\\[A-Za-z][^;]*;/g, '')
    .trim();
}

function polygonArea(pts: Array<{ x: number; y: number }>): number {
  if (pts.length < 3) return 0;
  let sum = 0;
  for (let i = 0; i < pts.length; i++) {
    const j = (i + 1) % pts.length;
    sum += pts[i].x * pts[j].y - pts[j].x * pts[i].y;
  }
  return Math.abs(sum) / 2;
}

function pointInPolygon(
  x: number,
  y: number,
  pts: Array<{ x: number; y: number }>,
): boolean {
  let inside = false;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const xi = pts[i].x;
    const yi = pts[i].y;
    const xj = pts[j].x;
    const yj = pts[j].y;
    const intersect =
      yi > y !== yj > y &&
      x < ((xj - xi) * (y - yi)) / (yj - yi + 1e-12) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}

function parseDxfEntityAttributes(
  entityType: string,
  attrs: Array<{ code: number; val: string }>,
  defaultLayer = '0',
): {
  layer: string;
  blockName?: string;
  insertTransform?: {
    x: number;
    y: number;
    sx: number;
    sy: number;
    rotDeg: number;
  };
  textInfo?: { x: number; y: number; text: string; height: number };
  closedPoly?: ClosedPoly;
  primitives: DxfPrimitive[];
} {
  let layer = defaultLayer;
  for (const a of attrs) {
    if (a.code === 8 && a.val) {
      layer = a.val;
      break;
    }
  }

  const getFirstNum = (c: number, fb = 0) => {
    const hit = attrs.find((a) => a.code === c);
    const n = hit ? parseFloat(hit.val) : fb;
    return Number.isFinite(n) ? n : fb;
  };
  const getFirstStr = (c: number) => attrs.find((a) => a.code === c)?.val ?? '';

  const primitives: DxfPrimitive[] = [];
  let closedPoly: ClosedPoly | undefined;
  let textInfo: { x: number; y: number; text: string; height: number } | undefined;

  if (entityType === 'LINE') {
    primitives.push({
      kind: 'line',
      layer,
      x1: getFirstNum(10),
      y1: getFirstNum(20),
      x2: getFirstNum(11),
      y2: getFirstNum(21),
    });
  } else if (entityType === 'LWPOLYLINE') {
    const xs = attrs
      .filter((a) => a.code === 10)
      .map((a) => parseFloat(a.val));
    const ys = attrs
      .filter((a) => a.code === 20)
      .map((a) => parseFloat(a.val));
    const flags = getFirstNum(70, 0);
    const len = Math.min(xs.length, ys.length);
    const pts: Array<{ x: number; y: number }> = [];
    for (let i = 0; i < len; i++) {
      if (Number.isFinite(xs[i]) && Number.isFinite(ys[i])) {
        pts.push({ x: xs[i], y: ys[i] });
      }
    }
    const closedByPts =
      pts.length > 2 &&
      Math.hypot(pts[0].x - pts[pts.length - 1].x, pts[0].y - pts[pts.length - 1].y) <
        1e-4;
    const isClosed = (flags & 1) === 1 || closedByPts;
    for (let i = 1; i < pts.length; i++) {
      primitives.push({
        kind: 'line',
        layer,
        x1: pts[i - 1].x,
        y1: pts[i - 1].y,
        x2: pts[i].x,
        y2: pts[i].y,
      });
    }
    if ((flags & 1) === 1 && pts.length > 2 && !closedByPts) {
      primitives.push({
        kind: 'line',
        layer,
        x1: pts[pts.length - 1].x,
        y1: pts[pts.length - 1].y,
        x2: pts[0].x,
        y2: pts[0].y,
      });
    }
    if (isClosed && pts.length >= 3) {
      const areaRaw = polygonArea(pts);
      if (areaRaw > 1e-4) {
        closedPoly = { layer, pts, areaRaw };
      }
    }
  } else if (entityType === 'CIRCLE') {
    primitives.push({
      kind: 'circle',
      layer,
      cx: getFirstNum(10),
      cy: getFirstNum(20),
      r: getFirstNum(40),
    });
  } else if (entityType === 'ARC') {
    primitives.push({
      kind: 'arc',
      layer,
      cx: getFirstNum(10),
      cy: getFirstNum(20),
      r: getFirstNum(40),
      startDeg: getFirstNum(50),
      endDeg: getFirstNum(51),
    });
  } else if (entityType === 'TEXT' || entityType === 'MTEXT') {
    const x = getFirstNum(10);
    const y = getFirstNum(20);
    const height = getFirstNum(40, 0.25);
    const raw =
      entityType === 'MTEXT'
        ? attrs
            .filter((a) => a.code === 3 || a.code === 1)
            .map((a) => a.val)
            .join('')
        : getFirstStr(1);
    const cleaned = cleanMText(raw);
    if (cleaned) {
      textInfo = { x, y, text: cleaned, height };
      primitives.push({ kind: 'text', layer, x, y, text: cleaned, height });
    }
  } else if (entityType === 'INSERT') {
    const blockName = getFirstStr(2);
    return {
      layer,
      blockName,
      insertTransform: {
        x: getFirstNum(10),
        y: getFirstNum(20),
        sx: getFirstNum(41, 1) || 1,
        sy: getFirstNum(42, 1) || 1,
        rotDeg: getFirstNum(50, 0),
      },
      primitives,
    };
  }

  return { layer, textInfo, closedPoly, primitives };
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
  let unitToMeters = 1;
  let inHeader = false;
  let inBlocks = false;
  let inEntities = false;
  let currentVar = '';
  let currentBlockName: string | null = null;

  const blockDefs = new Map<string, DxfPrimitive[]>();
  const layerCounts = new Map<string, number>();
  const entityCounts: Record<string, number> = {};
  const blockInsertCounts = new Map<string, number>();
  const textItems: Array<{ x: number; y: number; text: string; layer: string }> = [];
  const closedPolys: ClosedPoly[] = [];
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

  const trackPrimitive = (p: DxfPrimitive) => {
    primitives.push(p);
    if (p.kind === 'line') {
      expandBounds(p.x1, p.y1);
      expandBounds(p.x2, p.y2);
    } else if (p.kind === 'circle' || p.kind === 'arc') {
      expandBounds(p.cx - p.r, p.cy - p.r);
      expandBounds(p.cx + p.r, p.cy + p.r);
    } else if (p.kind === 'text') {
      expandBounds(p.x, p.y);
    }
  };

  let idx = 0;
  while (idx < pairs.length) {
    const { code, val } = pairs[idx];
    if (code === 0 && val === 'SECTION') {
      const next = pairs[idx + 1];
      if (next?.code === 2) {
        inHeader = next.val === 'HEADER';
        inBlocks = next.val === 'BLOCKS';
        inEntities = next.val === 'ENTITIES';
        idx += 2;
        continue;
      }
    }
    if (code === 0 && val === 'ENDSEC') {
      inHeader = false;
      inBlocks = false;
      inEntities = false;
      currentBlockName = null;
      idx++;
      continue;
    }

    if (inHeader) {
      if (code === 9) {
        currentVar = val;
      } else if (currentVar === '$INSUNITS' && code === 70) {
        const info = INSUNITS_MAP[parseInt(val, 10)];
        if (info) {
          units = info.label;
          unitToMeters = info.toMeters;
        }
      }
      idx++;
      continue;
    }

    if (inBlocks && code === 0) {
      const entityType = val;
      idx++;
      const attrs: Array<{ code: number; val: string }> = [];
      while (idx < pairs.length && pairs[idx].code !== 0) {
        attrs.push(pairs[idx]);
        idx++;
      }
      if (entityType === 'BLOCK') {
        const name = attrs.find((a) => a.code === 2)?.val ?? '';
        currentBlockName = name || null;
        if (currentBlockName && !blockDefs.has(currentBlockName)) {
          blockDefs.set(currentBlockName, []);
        }
      } else if (entityType === 'ENDBLK') {
        currentBlockName = null;
      } else if (currentBlockName) {
        const parsed = parseDxfEntityAttributes(entityType, attrs);
        const list = blockDefs.get(currentBlockName)!;
        list.push(...parsed.primitives);
      }
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

      if (
        entityType === 'ENDSEC' ||
        entityType === 'SEQEND' ||
        entityType === 'VERTEX'
      ) {
        continue;
      }

      entityCounts[entityType] = (entityCounts[entityType] ?? 0) + 1;
      const parsed = parseDxfEntityAttributes(entityType, attrs);
      layerCounts.set(parsed.layer, (layerCounts.get(parsed.layer) ?? 0) + 1);

      if (parsed.closedPoly) {
        closedPolys.push(parsed.closedPoly);
      }
      if (parsed.textInfo) {
        textItems.push({ ...parsed.textInfo, layer: parsed.layer });
      }

      if (entityType === 'INSERT' && parsed.blockName && parsed.insertTransform) {
        blockInsertCounts.set(
          parsed.blockName,
          (blockInsertCounts.get(parsed.blockName) ?? 0) + 1,
        );
        const def = blockDefs.get(parsed.blockName);
        const { x: tx, y: ty, sx, sy, rotDeg } = parsed.insertTransform;
        const rad = (rotDeg * Math.PI) / 180;
        const cosR = Math.cos(rad);
        const sinR = Math.sin(rad);
        const xform = (px: number, py: number) => {
          const lx = px * sx;
          const ly = py * sy;
          return {
            x: tx + lx * cosR - ly * sinR,
            y: ty + lx * sinR + ly * cosR,
          };
        };

        if (def && def.length > 0) {
          for (const bp of def) {
            const layer = bp.layer === '0' ? parsed.layer : bp.layer;
            if (bp.kind === 'line') {
              const p1 = xform(bp.x1, bp.y1);
              const p2 = xform(bp.x2, bp.y2);
              trackPrimitive({
                kind: 'line',
                layer,
                blockName: parsed.blockName,
                x1: p1.x,
                y1: p1.y,
                x2: p2.x,
                y2: p2.y,
              });
            } else if (bp.kind === 'circle') {
              const c = xform(bp.cx, bp.cy);
              trackPrimitive({
                kind: 'circle',
                layer,
                blockName: parsed.blockName,
                cx: c.x,
                cy: c.y,
                r: bp.r * Math.max(Math.abs(sx), Math.abs(sy)),
              });
            } else if (bp.kind === 'arc') {
              const c = xform(bp.cx, bp.cy);
              trackPrimitive({
                kind: 'arc',
                layer,
                blockName: parsed.blockName,
                cx: c.x,
                cy: c.y,
                r: bp.r * Math.max(Math.abs(sx), Math.abs(sy)),
                startDeg: bp.startDeg + rotDeg,
                endDeg: bp.endDeg + rotDeg,
              });
            }
          }
        } else {
          trackPrimitive({
            kind: 'circle',
            layer: parsed.layer,
            blockName: parsed.blockName,
            cx: tx,
            cy: ty,
            r: 0.15,
          });
        }
      } else {
        for (const p of parsed.primitives) {
          trackPrimitive(p);
        }
      }
      continue;
    }

    idx++;
  }

  // Deterministic room & apartment area calculation from closed polylines + contained text labels
  const areaScale = unitToMeters * unitToMeters;
  const unitPolys = closedPolys.filter((p) => {
    const l = p.layer.toLowerCase();
    return l.includes('unit') || l.includes('whg') || l.includes('apart');
  });
  const roomPolys = closedPolys.filter((p) => {
    const l = p.layer.toLowerCase();
    return (
      l.includes('room') ||
      l.includes('space') ||
      l.includes('raum') ||
      l.includes('zone') ||
      l.includes('area')
    );
  });

  const apartmentMeta = unitPolys.map((up, idxU) => {
    const insideTexts = textItems
      .filter((t) => pointInPolygon(t.x, t.y, up.pts))
      .map((t) => t.text);
    const unitLabel =
      insideTexts.find((t) => /whg|apt|unit|wohnung|app/i.test(t)) ??
      `Unit ${String.fromCharCode(65 + idxU)}`;
    return { poly: up, label: unitLabel };
  });

  const detectedRooms: Array<
    ParsedDxfRoom & { rawAreaSqM: number; pts: Array<{ x: number; y: number }> }
  > = [];
  for (const rp of roomPolys) {
    const rawAreaSqM = rp.areaRaw * areaScale;
    const areaSqM = Number(rawAreaSqM.toFixed(2));
    const inside = textItems.filter((t) => pointInPolygon(t.x, t.y, rp.pts));
    const codeText =
      inside.find((t) => /^\d+(\.\d+)?$/.test(t.text))?.text ?? '';
    const nameTexts = inside
      .filter((t) => t.text !== codeText && !/whg|wohnung/i.test(t.text))
      .map((t) => t.text);
    // Unnumbered and unlabelled tiny/structural polygons on ROOM_OUTLINE are shafts/voids per Swiss Dwellings standard
    const isUnnumberedVoid = !codeText && nameTexts.length === 0;
    const label = nameTexts.join(' / ') || codeText || 'Unlabelled Shaft / Void';
    const containingApt = apartmentMeta.find((a) =>
      inside.some((t) => pointInPolygon(t.x, t.y, a.poly.pts)),
    );
    const isExterior = EXTERIOR_ROOM_REGEX.test(label) || isUnnumberedVoid;
    detectedRooms.push({
      code: codeText,
      label,
      apartment: containingApt?.label ?? null,
      areaSqM,
      rawAreaSqM,
      pts: rp.pts,
      isExterior,
    });
  }

  const aptAgg = new Map<string, ParsedDxfApartment>();
  for (const a of apartmentMeta) {
    aptAgg.set(a.label, {
      name: a.label,
      indoorAreaSqM: 0,
      exteriorAreaSqM: 0,
      roomCount: 0,
    });
  }
  for (const r of detectedRooms) {
    if (!r.apartment) continue;
    const entry = aptAgg.get(r.apartment);
    if (!entry) continue;
    entry.roomCount += 1;
    if (r.isExterior) entry.exteriorAreaSqM += r.rawAreaSqM;
    else entry.indoorAreaSqM += r.rawAreaSqM;
  }
  const detectedApartments = Array.from(aptAgg.values()).map((a) => ({
    ...a,
    indoorAreaSqM: Number(a.indoorAreaSqM.toFixed(2)),
    exteriorAreaSqM: Number(a.exteriorAreaSqM.toFixed(2)),
  }));

  const indoorRooms = detectedRooms.filter((r) => !r.isExterior);
  const exteriorRooms = detectedRooms.filter((r) => r.isExterior);
  const sumIndoorRoomsSqM =
    indoorRooms.length > 0
      ? Number(
          indoorRooms.reduce((acc, r) => acc + r.rawAreaSqM, 0).toFixed(2),
        )
      : null;
  const sumExteriorSpacesSqM =
    exteriorRooms.length > 0
      ? Number(
          exteriorRooms.reduce((acc, r) => acc + r.rawAreaSqM, 0).toFixed(2),
        )
      : null;
  const cleanDetectedRooms: ParsedDxfRoom[] = detectedRooms.map(
    ({ rawAreaSqM: _raw, pts: _pts, ...rest }) => rest,
  );
  const roomPolygons: DxfRoomPolygon[] = detectedRooms.map(
    ({ rawAreaSqM: _raw, ...rest }) => rest,
  );

  const hasBounds =
    Number.isFinite(minX) &&
    Number.isFinite(maxX) &&
    maxX > minX &&
    maxY > minY;
  const width = hasBounds ? maxX - minX : 10;
  const height = hasBounds ? maxY - minY : 10;
  const pad = Math.max(width, height) * 0.06;

  const vbX = (hasBounds ? minX : 0) - pad;
  const vbY = -(hasBounds ? maxY : 10) - pad;
  const vbW = width + pad * 2;
  const vbH = height + pad * 2;
  const strokeW = Math.max(vbW, vbH) * 0.0022;

  const layerColor = (layer: string) => {
    const l = layer.toLowerCase();
    if (l.includes('wall') || l.includes('wand') || l.includes('struct'))
      return '#0f172a';
    if (l.includes('door') || l.includes('tuer')) return '#2563eb';
    if (l.includes('win') || l.includes('fenst')) return '#0284c7';
    if (l.includes('light') || l.includes('lamp') || l.includes('elec'))
      return '#d97706';
    if (l.includes('furn') || l.includes('fix')) return '#64748b';
    if (l.includes('room_outline')) return '#94a3b8';
    if (l.includes('unit_outline')) return '#475569';
    if (l.includes('text') || l.includes('room') || l.includes('label'))
      return '#1e293b';
    return '#334155';
  };

  const svgParts: string[] = [];
  const maxPrims = Math.min(primitives.length, 8000);
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
      const sweep = (p.endDeg - p.startDeg + 360) % 360;
      const largeArc = sweep > 180 ? 1 : 0;
      svgParts.push(
        `<path d="M ${x1.toFixed(3)} ${y1.toFixed(3)} A ${p.r.toFixed(3)} ${p.r.toFixed(3)} 0 ${largeArc} 0 ${x2.toFixed(3)} ${y2.toFixed(3)}" fill="none" stroke="${col}" stroke-width="${strokeW.toFixed(4)}"/>`,
      );
    } else if (p.kind === 'text') {
      const fontSize = Math.max(p.height * 0.85, strokeW * 5);
      const escaped = Array.from(p.text)
        .filter((ch) => {
          const code = ch.charCodeAt(0);
          return code === 9 || code === 10 || code === 13 || code >= 32;
        })
        .join('')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&apos;');
      svgParts.push(
        `<text x="${p.x.toFixed(3)}" y="${(-p.y).toFixed(3)}" fill="#0f172a" font-family="DM Sans, sans-serif" font-size="${fontSize.toFixed(4)}">${escaped}</text>`,
      );
    }
  }

  const safeVbX = Number.isFinite(vbX) ? vbX : 0;
  const safeVbY = Number.isFinite(vbY) ? vbY : 0;
  const safeVbW = Number.isFinite(vbW) && vbW > 0 ? vbW : 100;
  const safeVbH = Number.isFinite(vbH) && vbH > 0 ? vbH : 100;
  const svgMarkup = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="${safeVbX.toFixed(3)} ${safeVbY.toFixed(3)} ${safeVbW.toFixed(3)} ${safeVbH.toFixed(3)}" style="background:#ffffff">${svgParts.join('')}</svg>`;

  return {
    summary: {
      units,
      unitToMeters,
      layers: Array.from(layerCounts.entries())
        .map(([name, entityCount]) => ({ name, entityCount }))
        .sort((a, b) => b.entityCount - a.entityCount),
      entityCounts,
      blocks: Array.from(blockInsertCounts.entries())
        .map(([name, count]) => ({ name, count }))
        .sort((a, b) => b.count - a.count),
      texts: Array.from(new Set(textItems.map((t) => t.text))).slice(0, 400),
      detectedRooms: cleanDetectedRooms,
      detectedApartments,
      sumIndoorRoomsSqM,
      sumExteriorSpacesSqM,
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
    roomPolygons,
    viewBox: { x: vbX, y: vbY, width: vbW, height: vbH },
    svgMarkup,
  };
}
