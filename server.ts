import 'dotenv/config';
import express from 'express';
import type { NextFunction, Request, Response } from 'express';
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { GoogleGenAI, Type, ThinkingLevel } from '@google/genai';
import { createServer as createViteServer } from 'vite';
import { trainingEngine } from './server/trainingEngine.ts';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const ZIP_URL =
  'https://fwjydgxpttajiqpfmqtj.supabase.co/storage/v1/object/public/stoa-junction-2026/junction-dataset.zip';

interface StoredSample {
  id: string;
  title: string;
  source: string;
  format: 'png' | 'dxf' | 'ifc';
  zipPath: string;
  comp: number;
  comp_sz: number;
  uncomp_sz: number;
  loc_off: number;
  groundTruthQuestions: Array<{
    question: string;
    answer: string | number;
    unit: string | null;
    type: string;
  }>;
}

interface JunctionKnowledge {
  datasetStats: {
    totalFilesInZip: number;
    totalQuestions: number;
    uniqueAnnotatedDrawings: number;
    questionTypes: Record<string, number>;
    sourceDistribution: Record<string, number>;
  };
  exemplars: Record<
    string,
    Array<{
      path: string;
      question: string;
      answer: string | number;
      unit: string | null;
      type: string;
    }>
  >;
  samples: StoredSample[];
}

const knowledgePath = path.join(__dirname, 'server', 'junctionKnowledge.json');
const knowledge: JunctionKnowledge = JSON.parse(
  fs.readFileSync(knowledgePath, 'utf-8'),
);

const sampleBufferCache = new Map<string, Buffer>();

async function fetchZipEntry(sample: StoredSample): Promise<Buffer> {
  const cached = sampleBufferCache.get(sample.id);
  if (cached) return cached;

  const start = sample.loc_off;
  const end = start + 30 + 512 + sample.comp_sz - 1;
  const resp = await fetch(ZIP_URL, {
    headers: { Range: `bytes=${start}-${end}` },
  });
  if (!resp.ok && resp.status !== 206) {
    throw new Error(`Failed to stream sample from dataset (${resp.status})`);
  }
  const chunk = Buffer.from(await resp.arrayBuffer());
  const fnLen = chunk.readUInt16LE(26);
  const exLen = chunk.readUInt16LE(28);
  const dataStart = 30 + fnLen + exLen;
  const compressed = chunk.subarray(dataStart, dataStart + sample.comp_sz);

  const output =
    sample.comp === 8 ? zlib.inflateRawSync(compressed) : compressed;
  sampleBufferCache.set(sample.id, output);
  return output;
}

function formatStructuredSummaryForPrompt(
  summary: Record<string, unknown> | undefined,
): string {
  if (!summary || Object.keys(summary).length === 0) return '';
  const fmt = String(summary.format || '').toUpperCase();
  const lines: string[] = [];

  if (fmt === 'IFC') {
    lines.push(
      `FORMAT: IFC (${summary.schema ?? 'IFC'}) | Tool: ${summary.authoringTool ?? 'Unknown'} | Project: ${summary.projectName ?? 'Untitled'} | Native Unit: ${summary.lengthUnit ?? 'm'}`,
    );
    if (summary.boundingBoxMeters) {
      lines.push(`3D Bounding Box (m): ${JSON.stringify(summary.boundingBoxMeters)}`);
    }
    if (Array.isArray(summary.storeys) && summary.storeys.length > 0) {
      lines.push(`STOREY BREAKDOWN (${summary.storeys.length} storeys):`);
      for (const st of summary.storeys) {
        lines.push(`  - ${JSON.stringify(st)}`);
      }
    }
    if (summary.elementCounts) {
      lines.push(`ELEMENT COUNTS: ${JSON.stringify(summary.elementCounts)}`);
    }
    if (summary.wallStats) {
      lines.push(`WALL QUANTITIES: ${JSON.stringify(summary.wallStats)}`);
    }
    if (summary.slabStats) {
      lines.push(`SLAB QUANTITIES: ${JSON.stringify(summary.slabStats)}`);
    }
    if (Array.isArray(summary.doorTypes) && summary.doorTypes.length > 0) {
      lines.push(`DOOR TYPES: ${JSON.stringify(summary.doorTypes)}`);
    }
    if (Array.isArray(summary.doorInstances) && summary.doorInstances.length > 0) {
      lines.push(
        `DOOR INSTANCES (exact #id, type, storey, dimensions, adjacent spaces): ${JSON.stringify(summary.doorInstances.slice(0, 120))}`,
      );
    }
    if (Array.isArray(summary.windowTypes) && summary.windowTypes.length > 0) {
      lines.push(`WINDOW TYPES: ${JSON.stringify(summary.windowTypes)}`);
    }
    if (Array.isArray(summary.windowInstances) && summary.windowInstances.length > 0) {
      lines.push(
        `WINDOW INSTANCES (exact #id, type, storey, dimensions, adjacent spaces): ${JSON.stringify(summary.windowInstances.slice(0, 120))}`,
      );
    }
    if (Array.isArray(summary.lightingFixtures)) {
      lines.push(
        `LUMINAIRES / LIGHT FIXTURES (${summary.lightingFixtures.length} families): ${JSON.stringify(summary.lightingFixtures)}`,
      );
    }
    if (Array.isArray(summary.lightingInstances) && summary.lightingInstances.length > 0) {
      lines.push(
        `LIGHTING FIXTURE INSTANCES (exact #id, type, storey): ${JSON.stringify(summary.lightingInstances.slice(0, 120))}`,
      );
    }
    if (
      Array.isArray(summary.lightingControls) &&
      summary.lightingControls.length > 0
    ) {
      lines.push(
        `LIGHTING SWITCHES / CONTROLS (not luminaires): ${JSON.stringify(summary.lightingControls)}`,
      );
    }
    if (
      Array.isArray(summary.electricalDevices) &&
      summary.electricalDevices.length > 0
    ) {
      lines.push(
        `ELECTRICAL RECEPTACLES / PANELS: ${JSON.stringify(summary.electricalDevices)}`,
      );
    }
    if (
      Array.isArray(summary.sanitaryAndAppliances) &&
      summary.sanitaryAndAppliances.length > 0
    ) {
      lines.push(
        `SANITARY & APPLIANCES: ${JSON.stringify(summary.sanitaryAndAppliances)}`,
      );
    }
    if (
      Array.isArray(summary.furnitureAndFixtures) &&
      summary.furnitureAndFixtures.length > 0
    ) {
      lines.push(
        `FURNITURE & FIXTURES: ${JSON.stringify(summary.furnitureAndFixtures)}`,
      );
    }
    if (Array.isArray(summary.spaces) && summary.spaces.length > 0) {
      lines.push(
        `IFCSPACE ROOMS (totalSpaceAreaSqM=${summary.totalSpaceAreaSqM ?? 'unknown'}, count=${summary.spaces.length}):`,
      );
      for (const sp of summary.spaces.slice(0, 120)) {
        lines.push(`  - ${JSON.stringify(sp)}`);
      }
      if (summary.spaces.length > 120) {
        lines.push(`  ... (${summary.spaces.length - 120} more spaces omitted)`);
      }
    }
    return lines.join('\n');
  }

  if (fmt === 'DXF') {
    lines.push(
      `FORMAT: CAD DXF | Units: ${summary.units ?? 'Metres'} | Bounding Box: ${JSON.stringify(summary.boundingBox)}`,
    );
    lines.push(`LAYERS: ${JSON.stringify(summary.layers)}`);
    lines.push(`ENTITY COUNTS: ${JSON.stringify(summary.entityCounts)}`);
    if (Array.isArray(summary.blocks) && summary.blocks.length > 0) {
      lines.push(`BLOCK INSERTS: ${JSON.stringify(summary.blocks)}`);
    }
    if (
      Array.isArray(summary.detectedApartments) &&
      summary.detectedApartments.length > 0
    ) {
      lines.push(
        `DETERMINISTIC APARTMENT POLYGON SUMS: ${JSON.stringify(summary.detectedApartments)}`,
      );
    }
    if (Array.isArray(summary.detectedRooms) && summary.detectedRooms.length > 0) {
      lines.push(
        `DETERMINISTIC ROOM POLYGONS (sumIndoorRoomsSqM=${summary.sumIndoorRoomsSqM}, sumExteriorSpacesSqM=${summary.sumExteriorSpacesSqM}):`,
      );
      for (const rm of summary.detectedRooms.slice(0, 100)) {
        lines.push(`  - ${JSON.stringify(rm)}`);
      }
    }
    if (Array.isArray(summary.texts) && summary.texts.length > 0) {
      lines.push(`CAD TEXT LABELS: ${JSON.stringify(summary.texts.slice(0, 200))}`);
    }
    return lines.join('\n');
  }

  return JSON.stringify(summary, null, 2).slice(0, 32000);
}

const SQM_TO_SQFT = 10.7639104;

function round2(n: number): number {
  return Math.round(n * 100) / 100;
}

function toFiniteOrNull(val: unknown): number | null {
  return typeof val === 'number' && Number.isFinite(val) ? val : null;
}

function parseJsonObjectSafe(rawInput: string): Record<string, unknown> {
  const unfenced = rawInput
    .trim()
    .replace(/^```(?:json)?\s*/i, '')
    .replace(/\s*```$/i, '')
    .trim();
  try {
    const parsed = JSON.parse(unfenced);
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>;
    }
  } catch {
    const first = unfenced.indexOf('{');
    const last = unfenced.lastIndexOf('}');
    if (first !== -1 && last > first) {
      const parsed = JSON.parse(unfenced.slice(first, last + 1));
      if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
        return parsed as Record<string, unknown>;
      }
    }
  }
  return {};
}

function reconcileAndVerifyAnalysis(
  rawAnalysis: Record<string, unknown>,
  fileKind: string,
): Record<string, unknown> {
  const analysis = structuredClone(rawAnalysis) as {
    sceneType?: string;
    summary?: string;
    scaleAndCalibration?: string;
    dimensions?: {
      totalAreaSqM?: number | null;
      totalAreaSqFt?: number | null;
      totalAreaEvidence?: string;
      printedLivingAreaSqM?: number | null;
      printedTotalAreaSqM?: number | null;
      sumInteriorRoomsSqM?: number | null;
      sumExteriorSpacesSqM?: number | null;
      calculatedInteriorPlusExteriorSqM?: number | null;
      grossFootprintAreaSqM?: number | null;
      grossFootprintEvidence?: string;
      footprintDimensionsText?: string;
      estimatedCeilingHeightM?: number | null;
      ceilingHeightEvidence?: string;
      estimatedVolumeCuM?: number | null;
      volumeEvidence?: string;
      footprintSummary?: string;
    };
    rooms?: Array<{
      name?: string;
      labelInDrawing?: string;
      isExteriorOrPorch?: boolean;
      estimatedAreaSqM?: number | null;
      estimatedAreaSqFt?: number | null;
      areaEvidence?: string;
      dimensionsText?: string;
      dimensionsEvidence?: string;
      notes?: string;
      boxes2D?: unknown[];
    }>;
    detectedObjects?: Array<{
      category?: string;
      name?: string;
      count?: number;
      unit?: string;
      evidence?: string;
      confidence?: string;
      locationsOrNotes?: string;
      boxes2D?: unknown[];
    }>;
    discrepanciesAndUncertainties?: string[];
    datasetPatternsApplied?: string[];
  };

  const rawRooms = Array.isArray(analysis.rooms) ? analysis.rooms : [];
  const rawObjects = Array.isArray(analysis.detectedObjects)
    ? analysis.detectedObjects
    : [];
  const rawDims = analysis.dimensions ?? {};

  const dims = {
    totalAreaSqM: toFiniteOrNull(rawDims.totalAreaSqM),
    totalAreaSqFt: toFiniteOrNull(rawDims.totalAreaSqFt),
    totalAreaEvidence: String(rawDims.totalAreaEvidence || 'Unknown'),
    printedLivingAreaSqM: toFiniteOrNull(rawDims.printedLivingAreaSqM),
    printedTotalAreaSqM: toFiniteOrNull(rawDims.printedTotalAreaSqM),
    sumInteriorRoomsSqM: toFiniteOrNull(rawDims.sumInteriorRoomsSqM),
    sumExteriorSpacesSqM: toFiniteOrNull(rawDims.sumExteriorSpacesSqM),
    calculatedInteriorPlusExteriorSqM: toFiniteOrNull(
      rawDims.calculatedInteriorPlusExteriorSqM,
    ),
    grossFootprintAreaSqM: toFiniteOrNull(rawDims.grossFootprintAreaSqM),
    grossFootprintEvidence: String(rawDims.grossFootprintEvidence || 'Unknown'),
    footprintDimensionsText: String(
      rawDims.footprintDimensionsText || 'Not printed',
    ),
    estimatedCeilingHeightM: toFiniteOrNull(rawDims.estimatedCeilingHeightM),
    ceilingHeightEvidence: String(rawDims.ceilingHeightEvidence || 'Unknown'),
    estimatedVolumeCuM: toFiniteOrNull(rawDims.estimatedVolumeCuM),
    volumeEvidence: String(rawDims.volumeEvidence || 'Unknown'),
    footprintSummary: String(rawDims.footprintSummary || ''),
  };

  const discrepancies = new Set<string>(
    Array.isArray(analysis.discrepanciesAndUncertainties)
      ? analysis.discrepanciesAndUncertainties
          .filter((item): item is string => typeof item === 'string' && Boolean(item.trim()))
      : [],
  );

  // Sync m² <-> ft² per room and guarantee non-undefined fields
  let interiorSum = 0;
  let hasInterior = false;
  let exteriorSum = 0;
  let hasExterior = false;

  const rooms = rawRooms.map((r, idx) => {
    let sqM = toFiniteOrNull(r?.estimatedAreaSqM);
    let sqFt = toFiniteOrNull(r?.estimatedAreaSqFt);
    if (sqM !== null) {
      sqM = round2(sqM);
      sqFt = round2(sqM * SQM_TO_SQFT);
    } else if (sqFt !== null) {
      sqM = round2(sqFt / SQM_TO_SQFT);
      sqFt = round2(sqFt);
    }
    const isExt = Boolean(r?.isExteriorOrPorch);
    if (sqM !== null) {
      if (isExt) {
        exteriorSum += sqM;
        hasExterior = true;
      } else {
        interiorSum += sqM;
        hasInterior = true;
      }
    }
    const normRoomBoxes = normalizeBoxes2D(r?.boxes2D);
    return {
      name: String(r?.name || `Space ${idx + 1}`),
      labelInDrawing: String(r?.labelInDrawing || r?.name || ''),
      isExteriorOrPorch: isExt,
      estimatedAreaSqM: sqM,
      estimatedAreaSqFt: sqFt,
      areaEvidence: String(r?.areaEvidence || 'Unknown'),
      dimensionsText: String(r?.dimensionsText || 'Not printed'),
      dimensionsEvidence: String(r?.dimensionsEvidence || 'Unknown'),
      notes: String(r?.notes || ''),
      ...(normRoomBoxes.length > 0 ? { boxes2D: normRoomBoxes } : {}),
    };
  });

  const detectedObjects = rawObjects.map((o, idx) => {
    const count =
      typeof o?.count === 'number' && Number.isFinite(o.count)
        ? Math.max(0, Math.round(o.count))
        : 0;
    const normObjBoxes = normalizeBoxes2D(
      o?.boxes2D,
      count > 0 ? count : undefined,
    );
    return {
      category: String(o?.category || 'Other'),
      name: String(o?.name || `Object ${idx + 1}`),
      count,
      unit: String(o?.unit || 'count'),
      evidence: String(o?.evidence || 'Estimated'),
      confidence: String(o?.confidence || 'Medium'),
      locationsOrNotes: String(o?.locationsOrNotes || ''),
      ...(normObjBoxes.length > 0 ? { boxes2D: normObjBoxes } : {}),
    };
  });

  if (dims.printedLivingAreaSqM !== null) {
    dims.printedLivingAreaSqM = round2(dims.printedLivingAreaSqM);
  }
  if (dims.printedTotalAreaSqM !== null) {
    dims.printedTotalAreaSqM = round2(dims.printedTotalAreaSqM);
  }
  if (dims.grossFootprintAreaSqM !== null) {
    dims.grossFootprintAreaSqM = round2(dims.grossFootprintAreaSqM);
  }

  if (hasInterior) {
    dims.sumInteriorRoomsSqM = round2(interiorSum);
  }
  if (hasExterior) {
    dims.sumExteriorSpacesSqM = round2(exteriorSum);
  }
  if (hasInterior || hasExterior) {
    dims.calculatedInteriorPlusExteriorSqM = round2(
      (hasInterior ? interiorSum : 0) + (hasExterior ? exteriorSum : 0),
    );
  }

  // Flag printed vs summed discrepancies automatically if > 0.05 m²
  if (
    dims.printedLivingAreaSqM !== null &&
    dims.sumInteriorRoomsSqM !== null
  ) {
    const diff = Math.abs(dims.printedLivingAreaSqM - dims.sumInteriorRoomsSqM);
    if (diff > 0.05) {
      discrepancies.add(
        `Arithmetic discrepancy of ${diff.toFixed(2)} m² between printed living area summary (${dims.printedLivingAreaSqM.toFixed(2)} m²) and sum of labelled interior rooms (${dims.sumInteriorRoomsSqM.toFixed(2)} m²). Both values preserved.`,
      );
    }
  }
  if (
    dims.printedTotalAreaSqM !== null &&
    dims.calculatedInteriorPlusExteriorSqM !== null
  ) {
    const diff = Math.abs(
      dims.printedTotalAreaSqM - dims.calculatedInteriorPlusExteriorSqM,
    );
    if (diff > 0.05) {
      discrepancies.add(
        `Arithmetic discrepancy of ${diff.toFixed(2)} m² between printed total area (${dims.printedTotalAreaSqM.toFixed(2)} m²) and summed interior + exterior spaces (${dims.calculatedInteriorPlusExteriorSqM.toFixed(2)} m²). Both values preserved.`,
      );
    }
  }

  // Strictly enforce no fabricated ceiling height / volume on 2D drawings
  const is2D = fileKind.startsWith('2d-');
  if (
    dims.ceilingHeightEvidence === 'Unknown' ||
    dims.estimatedCeilingHeightM === null ||
    (is2D && dims.ceilingHeightEvidence === 'Estimated')
  ) {
    dims.estimatedCeilingHeightM = null;
    dims.ceilingHeightEvidence = 'Unknown';
    dims.estimatedVolumeCuM = null;
    dims.volumeEvidence = 'Unknown';
  } else {
    dims.estimatedCeilingHeightM = round2(dims.estimatedCeilingHeightM);
    if (dims.estimatedVolumeCuM !== null) {
      dims.estimatedVolumeCuM = round2(dims.estimatedVolumeCuM);
    }
  }

  // Ensure totalAreaSqFt matches totalAreaSqM
  if (dims.totalAreaSqM !== null) {
    dims.totalAreaSqM = round2(dims.totalAreaSqM);
    dims.totalAreaSqFt = round2(dims.totalAreaSqM * SQM_TO_SQFT);
  } else if (dims.totalAreaSqFt !== null) {
    dims.totalAreaSqM = round2(dims.totalAreaSqFt / SQM_TO_SQFT);
    dims.totalAreaSqFt = round2(dims.totalAreaSqFt);
  }

  return {
    sceneType: String(analysis.sceneType || 'Architectural Drawing / Model'),
    summary: String(analysis.summary || 'Spatial takeoff completed.'),
    scaleAndCalibration: String(analysis.scaleAndCalibration || ''),
    dimensions: dims,
    rooms,
    detectedObjects,
    discrepanciesAndUncertainties: Array.from(discrepancies),
    datasetPatternsApplied: Array.isArray(analysis.datasetPatternsApplied)
      ? analysis.datasetPatternsApplied.filter(
          (p): p is string => typeof p === 'string',
        )
      : [],
  };
}

interface NormalizedBox2D {
  ymin: number;
  xmin: number;
  ymax: number;
  xmax: number;
  label?: string;
}

function normalizeBoxes2D(
  rawList: unknown,
  expectedCount?: number,
  allowFullCanvas = false,
): NormalizedBox2D[] {
  if (!Array.isArray(rawList)) return [];
  const candidates: NormalizedBox2D[] = [];

  for (const item of rawList) {
    if (!item || typeof item !== 'object') continue;
    const rec = item as Record<string, unknown>;
    let y1 = NaN;
    let x1 = NaN;
    let y2 = NaN;
    let x2 = NaN;

    const arr = Array.isArray(rec.box_2d)
      ? rec.box_2d
      : Array.isArray(rec.box2d)
        ? rec.box2d
        : Array.isArray(item)
          ? item
          : null;

    if (arr && arr.length >= 4) {
      y1 = Number(arr[0]);
      x1 = Number(arr[1]);
      y2 = Number(arr[2]);
      x2 = Number(arr[3]);
    } else {
      y1 = Number(rec.ymin);
      x1 = Number(rec.xmin);
      y2 = Number(rec.ymax);
      x2 = Number(rec.xmax);
    }

    if (
      !Number.isFinite(y1) ||
      !Number.isFinite(x1) ||
      !Number.isFinite(y2) ||
      !Number.isFinite(x2)
    ) {
      continue;
    }

    // Handle [0..1] normalized floats if returned instead of [0..1000]
    const maxAbs = Math.max(Math.abs(y1), Math.abs(x1), Math.abs(y2), Math.abs(x2));
    if (maxAbs > 0 && maxAbs <= 1.05) {
      y1 *= 1000;
      x1 *= 1000;
      y2 *= 1000;
      x2 *= 1000;
    }

    let ymin = Math.round(Math.max(0, Math.min(1000, Math.min(y1, y2))));
    let ymax = Math.round(Math.max(0, Math.min(1000, Math.max(y1, y2))));
    let xmin = Math.round(Math.max(0, Math.min(1000, Math.min(x1, x2))));
    let xmax = Math.round(Math.max(0, Math.min(1000, Math.max(x1, x2))));

    // Ensure minimum visible span for thin wall openings or small icons without puffing onto walls
    if (ymax - ymin < 10) {
      const cy = Math.round((ymin + ymax) / 2);
      ymin = Math.max(0, cy - 5);
      ymax = Math.min(1000, cy + 5);
    }
    if (xmax - xmin < 10) {
      const cx = Math.round((xmin + xmax) / 2);
      xmin = Math.max(0, cx - 5);
      xmax = Math.min(1000, cx + 5);
    }

    const w = xmax - xmin;
    const h = ymax - ymin;
    if (w < 6 || h < 6) continue;
    // Reject accidental whole-sheet bounding boxes
    if (!allowFullCanvas && w > 965 && h > 965) continue;

    const label =
      typeof rec.label === 'string' && rec.label.trim()
        ? rec.label.trim()
        : undefined;

    // Deduplicate only near-identical boxes (IoU > 0.92) so adjacent chairs/doors are preserved
    const isDuplicate = candidates.some((existing) => {
      const ix1 = Math.max(existing.xmin, xmin);
      const iy1 = Math.max(existing.ymin, ymin);
      const ix2 = Math.min(existing.xmax, xmax);
      const iy2 = Math.min(existing.ymax, ymax);
      if (ix2 <= ix1 || iy2 <= iy1) return false;
      const inter = (ix2 - ix1) * (iy2 - iy1);
      const areaA = (existing.xmax - existing.xmin) * (existing.ymax - existing.ymin);
      const areaB = w * h;
      const iou = inter / Math.max(1, areaA + areaB - inter);
      return iou > 0.92;
    });

    if (!isDuplicate) {
      candidates.push({
        ymin,
        xmin,
        ymax,
        xmax,
        ...(label ? { label } : {}),
      });
    }
  }

  if (expectedCount && expectedCount > 0 && candidates.length > expectedCount) {
    return candidates.slice(0, expectedCount);
  }
  return candidates;
}

const BOX_2D_ITEMS_SCHEMA = {
  type: Type.OBJECT,
  properties: {
    box_2d: {
      type: Type.ARRAY,
      items: { type: Type.INTEGER },
      description:
        'Tight 4-integer bounding box [ymin, xmin, ymax, xmax] normalized to 0..1000 hugging ONLY the symbol or room.',
    },
    label: {
      type: Type.STRING,
      description: 'Short label for this specific symbol or room instance.',
    },
  },
  required: ['box_2d'],
  propertyOrdering: ['box_2d', 'label'],
};

const DETECT_2D_INDEX_SCHEMA = {
  type: Type.OBJECT,
  properties: {
    elements: {
      type: Type.ARRAY,
      items: {
        type: Type.OBJECT,
        properties: {
          box_2d: {
            type: Type.ARRAY,
            items: { type: Type.INTEGER },
            description:
              'Tight 4-integer bounding box [ymin, xmin, ymax, xmax] normalized to 0..1000.',
          },
          category: {
            type: Type.STRING,
            description:
              'One of: "Door", "Window", "Furniture", "Plumbing", "Appliance", "Lighting", "Space".',
          },
          label: {
            type: Type.STRING,
            description:
              'Specific name, e.g. "interior door", "entrance door", "sliding door", "window", "bed", "sofa", "dining table", "chair", "wardrobe", "toilet", "sink", "shower", "bathtub", "stove", "lamp", or exact room name.',
          },
        },
        required: ['box_2d', 'category', 'label'],
        propertyOrdering: ['box_2d', 'category', 'label'],
      },
    },
  },
  required: ['elements'],
};

interface SpatialTileInput {
  id: string;
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  mimeType: string;
  dataBase64: string;
}

interface SpatialIndex2DItem {
  ymin: number;
  xmin: number;
  ymax: number;
  xmax: number;
  category: string;
  label: string;
}

interface RawCandidateBox extends SpatialIndex2DItem {
  fromTile: boolean;
}

const MAX_PARALLEL_LABEL_REQUESTS = 50;

const spatialIndexCache = new Map<string, SpatialIndex2DItem[]>();
const spatialIndexInFlight = new Map<string, Promise<SpatialIndex2DItem[]>>();
const locate2DCache = new Map<string, NormalizedBox2D[]>();

function hashPayloadKey(...parts: Array<string | number | undefined>): string {
  const h = crypto.createHash('sha1');
  for (const p of parts) {
    h.update(String(p ?? ''));
    h.update('|');
  }
  return h.digest('hex');
}

interface ResolvedRoboticsEndpoint {
  client: GoogleGenAI;
  model: string;
  apiVersion?: string;
}

let cachedRoboticsEndpoint: ResolvedRoboticsEndpoint | null = null;
let inflightRoboticsProbe: Promise<ResolvedRoboticsEndpoint> | null = null;

async function resolveRoboticsLabeler(): Promise<ResolvedRoboticsEndpoint> {
  if (cachedRoboticsEndpoint) return cachedRoboticsEndpoint;
  if (inflightRoboticsProbe) return inflightRoboticsProbe;

  inflightRoboticsProbe = (async (): Promise<ResolvedRoboticsEndpoint> => {
    const candidates: Array<{ model: string; apiVersion?: string }> = [
      { model: 'gemini-robotics-er-2-preview', apiVersion: 'v1beta' },
      { model: 'gemini-robotics-er-2-preview', apiVersion: undefined },
      { model: 'gemini-robotics-er-2.0-preview', apiVersion: 'v1beta' },
      { model: 'gemini-robotics-er-1.5-preview', apiVersion: 'v1beta' },
      { model: 'gemini-4-carbon', apiVersion: undefined },
    ];

    for (const cand of candidates) {
      try {
        const client = getAiClient(cand.apiVersion);
        await client.models.generateContent({
          model: cand.model,
          contents: 'Return {"ok":true}',
          config: {
            thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
            responseMimeType: 'application/json',
          },
        });
        const resolved: ResolvedRoboticsEndpoint = {
          client,
          model: cand.model,
          apiVersion: cand.apiVersion,
        };
        cachedRoboticsEndpoint = resolved;
        console.log(
          `[2D Labeling] Resolved spatial reasoning model: ${cand.model} (${cand.apiVersion ?? 'default'})`,
        );
        return resolved;
      } catch {
        // Try next candidate
      }
    }

    const fallback: ResolvedRoboticsEndpoint = {
      client: getAiClient(),
      model: 'gemini-4-hydrogen',
    };
    cachedRoboticsEndpoint = fallback;
    return fallback;
  })();

  try {
    return await inflightRoboticsProbe;
  } finally {
    inflightRoboticsProbe = null;
  }
}

async function runWithConcurrencyLimit<T>(
  tasks: Array<() => Promise<T>>,
  maxConcurrency = MAX_PARALLEL_LABEL_REQUESTS,
): Promise<T[]> {
  if (tasks.length === 0) return [];
  const results = new Array<T>(tasks.length);
  let nextIdx = 0;
  const workerCount = Math.min(maxConcurrency, tasks.length);

  const workers = Array.from({ length: workerCount }, async () => {
    while (true) {
      const idx = nextIdx++;
      if (idx >= tasks.length) break;
      results[idx] = await tasks[idx]();
    }
  });

  await Promise.all(workers);
  return results;
}

async function callRoboticsWithRetry<T>(fn: () => Promise<T>): Promise<T> {
  let lastErr: unknown;
  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      return await fn();
    } catch (err) {
      lastErr = err;
      const msg = err instanceof Error ? err.message : String(err);
      if (attempt === 0 && /(429|503|rate|quota|overloaded|UNAVAILABLE)/i.test(msg)) {
        const backoffMs = 280 + Math.floor(Math.random() * 380);
        await new Promise((r) => setTimeout(r, backoffMs));
        continue;
      }
      throw err;
    }
  }
  throw lastErr;
}

interface SpecialistSpec {
  id: string;
  category:
    | 'Door'
    | 'Window'
    | 'Furniture'
    | 'Plumbing'
    | 'Appliance'
    | 'Lighting'
    | 'Space';
  tileAllowed: boolean;
  prompt: string;
}

const SPECIALIST_SPECS: SpecialistSpec[] = [
  {
    id: 'doors',
    category: 'Door',
    tileAllowed: true,
    prompt:
      'Detect ONLY true Door symbols in this blueprint view: hinged doors (straight thin door leaf + quarter-circle swing arc bridging a wall opening), double swing doors (two meeting leaves/arcs as 1 assembly), sliding/pocket doors, or bi-fold closet doors. Stop box_2d strictly at hinge jamb and latch jamb—do NOT include solid wall segments. STRICTLY EXCLUDE curved corner showers inside bathrooms, open wall gaps/archways without a door leaf, and stair arrows. Emit category "Door" and label ("interior door", "entrance door", "sliding door", "balcony door", or "double door"). Return {"elements":[]} if none are visible.',
  },
  {
    id: 'windows',
    category: 'Window',
    tileAllowed: true,
    prompt:
      'Detect ONLY Window glazing symbols embedded inside exterior or partition walls in this blueprint view (2-to-4 thin parallel glazing lines spanning jamb-to-jamb inside the wall thickness). Tightly hug ONLY the narrow glazing band, excluding solid wall segments between windows, wardrobes/cabinets, radiators, balcony railings, and hinged doors with swing arcs. Emit category "Window" and label "window". Return {"elements":[]} if none are visible.',
  },
  {
    id: 'furniture-seating-tables',
    category: 'Furniture',
    tileAllowed: true,
    prompt:
      'Detect ONLY Seating & Table furniture symbols in this blueprint view: each individual "chair", "dining chair", "armchair", "stool", "sofa", "dining table", "coffee table", "desk", or "kitchen island". CRITICAL: Emit one tight separate box for EACH individual chair (even chairs partially tucked around a dining table) and one separate box for the tabletop — NEVER merge a table and chairs into one combined box. Emit category "Furniture". Return {"elements":[]} if none are visible.',
  },
  {
    id: 'furniture-beds-storage',
    category: 'Furniture',
    tileAllowed: true,
    prompt:
      'Detect ONLY Bed & Storage furniture symbols in this blueprint view: each individual "bed", "double bed", "single bed", "crib", "nightstand", "wardrobe", "closet", "cabinet", "dresser", or "bookshelf" (including 60 cm storage modules with X cross, diagonal rod line, or SK/VH/KAAPPI/Schrank tags). Keep nightstands separate from beds, and NEVER mislabel wooden sauna benches (room S) or stair treads as furniture. Emit category "Furniture". Return {"elements":[]} if none are visible.',
  },
  {
    id: 'plumbing',
    category: 'Plumbing',
    tileAllowed: true,
    prompt:
      'Detect ONLY Sanitary & Plumbing fixture symbols in this blueprint view: each individual "toilet" (compact wall cistern + oval bowl, NOT the whole WC room), "sink" (basin bowl with drain/faucet only, NOT the bare kitchen counter), "shower" (square/corner tray with drain diagonals/circle), "bathtub" (oval tub in rim), or "bidet". Tightly hug ONLY each fixture contour. Emit category "Plumbing". Return {"elements":[]} if none are visible.',
  },
  {
    id: 'appliances',
    category: 'Appliance',
    tileAllowed: true,
    prompt:
      'Detect ONLY Kitchen & Laundry Appliance symbols in this blueprint view: each individual "stove" (60x60 cm square with 4 circular burner rings or tagged L/Liesi/Herd), "oven", "refrigerator" (tagged JK/PK/K/F), "dishwasher" (tagged APK/GS), "washing machine" (tagged PPK/PK/KR/WM or drum circle in bath/utility), or "dryer". Tightly hug ONLY each appliance module. Emit category "Appliance". Return {"elements":[]} if none are visible.',
  },
  {
    id: 'lighting',
    category: 'Lighting',
    tileAllowed: true,
    prompt:
      'Detect ONLY true Lighting / Luminaire symbols in this blueprint view: each individual "lamp", "ceiling light", "pendant light", "sconce", or "chandelier". STRICTLY EXCLUDE wall switches (small circle on wall with angled tick), electrical wall outlets, bathroom floor drains, and 4-ring stove burners. Emit category "Lighting". Return {"elements":[]} if none are visible.',
  },
  {
    id: 'spaces',
    category: 'Space',
    tileAllowed: false,
    prompt:
      'Detect every individual Room / Space interior boundary in this floor plan. Hug the INNER wall faces of each room. IMPORTANT: If any room, hallway (ET/Korridor), or open living-kitchen (OH+KT) is L-shaped, T-shaped, or U-shaped, decompose it into 2 or 3 tight non-overlapping interior rectangles with the SAME exact room label so adjacent rooms tucked into corners are NEVER overlapped. Emit category "Space".',
  },
];

function inferSubFamilyKey(category: string, label: string): string {
  const low = label.toLowerCase();
  if (category === 'Space') return `space:${low}`;
  if (category === 'Door' || low.includes('door')) return 'door';
  if (category === 'Window' || low.includes('window')) return 'window';
  if (low.includes('chair') || low.includes('stool') || low.includes('armchair')) {
    return 'chair';
  }
  if (low.includes('sofa') || low.includes('couch') || low.includes('sectional')) {
    return 'sofa';
  }
  if (low.includes('table') || low.includes('desk') || low.includes('island')) {
    return 'table';
  }
  if (low.includes('bed') || low.includes('crib')) return 'bed';
  if (low.includes('nightstand')) return 'nightstand';
  if (
    low.includes('wardrobe') ||
    low.includes('closet') ||
    low.includes('cabinet') ||
    low.includes('dresser') ||
    low.includes('shelf')
  ) {
    return 'storage';
  }
  if (low.includes('toilet') || low.includes('wc')) return 'toilet';
  if (low.includes('sink') || low.includes('basin')) return 'sink';
  if (low.includes('shower')) return 'shower';
  if (low.includes('bath') || low.includes('tub')) return 'bathtub';
  if (low.includes('stove') || low.includes('cooktop') || low.includes('oven')) {
    return 'stove';
  }
  if (low.includes('fridge') || low.includes('refrigerator')) return 'fridge';
  if (low.includes('wash') || low.includes('dryer')) return 'laundry';
  if (category === 'Lighting') return 'lighting';
  return `${category.toLowerCase()}:${low}`;
}

function fuseParallelSpatialDetections(
  candidates: RawCandidateBox[],
): SpatialIndex2DItem[] {
  // Filter invalid specks & oversized non-room boxes first
  const valid = candidates.filter((cand) => {
    const w = cand.xmax - cand.xmin;
    const h = cand.ymax - cand.ymin;
    if (w < 6 || h < 6) return false;
    if (cand.category !== 'Space' && w * h > 340000) return false;
    return true;
  });

  // Sort high-DPI tile predictions ahead of full-sheet predictions, and medium/tight symbols ahead of loose boxes
  const sorted = [...valid].sort((a, b) => {
    if (a.fromTile !== b.fromTile) return a.fromTile ? -1 : 1;
    const areaA = (a.xmax - a.xmin) * (a.ymax - a.ymin);
    const areaB = (b.xmax - b.xmin) * (b.ymax - b.ymin);
    return areaA - areaB;
  });

  interface BoxCluster {
    category: string;
    fam: string;
    label: string;
    members: Array<{ box: RawCandidateBox; weight: number }>;
    ymin: number;
    xmin: number;
    ymax: number;
    xmax: number;
    totalWeight: number;
  }

  const clusters: BoxCluster[] = [];

  const recomputeClusterConsensus = (cl: BoxCluster) => {
    let wSum = 0;
    let y0 = 0;
    let x0 = 0;
    let y1 = 0;
    let x1 = 0;
    for (const m of cl.members) {
      wSum += m.weight;
      y0 += m.box.ymin * m.weight;
      x0 += m.box.xmin * m.weight;
      y1 += m.box.ymax * m.weight;
      x1 += m.box.xmax * m.weight;
    }
    if (wSum > 0) {
      cl.ymin = Math.round(y0 / wSum);
      cl.xmin = Math.round(x0 / wSum);
      cl.ymax = Math.round(y1 / wSum);
      cl.xmax = Math.round(x1 / wSum);
      cl.totalWeight = wSum;
    }
  };

  for (const cand of sorted) {
    const famC = inferSubFamilyKey(cand.category, cand.label);
    const areaC = (cand.xmax - cand.xmin) * (cand.ymax - cand.ymin);
    const weight = cand.fromTile ? 1.65 : 1.0;

    let bestCluster: BoxCluster | null = null;
    let bestIoU = 0;

    for (const cl of clusters) {
      if ((cl.category === 'Space') !== (cand.category === 'Space')) continue;

      const ix1 = Math.max(cl.xmin, cand.xmin);
      const iy1 = Math.max(cl.ymin, cand.ymin);
      const ix2 = Math.min(cl.xmax, cand.xmax);
      const iy2 = Math.min(cl.ymax, cand.ymax);
      if (ix2 <= ix1 || iy2 <= iy1) continue;

      const inter = (ix2 - ix1) * (iy2 - iy1);
      const areaE = (cl.xmax - cl.xmin) * (cl.ymax - cl.ymin);
      const iou = inter / Math.max(1, areaE + areaC - inter);
      const minContain = inter / Math.max(1, Math.min(areaE, areaC));
      const areaRatio =
        Math.max(areaE, areaC) / Math.max(1, Math.min(areaE, areaC));

      let matchesCluster = false;
      if (cand.category === 'Space') {
        // Keep decomposed L-shaped sub-rectangles separate unless they substantially overlap
        matchesCluster = cl.fam === famC ? iou > 0.52 : iou > 0.72;
      } else if (cl.fam === famC) {
        matchesCluster = iou > 0.36 || (minContain > 0.78 && areaRatio < 2.5);
      } else if (cl.category === cand.category) {
        matchesCluster = iou > 0.68;
      } else {
        matchesCluster = iou > 0.82;
      }

      if (matchesCluster && iou >= bestIoU) {
        bestIoU = iou;
        bestCluster = cl;
      }
    }

    if (bestCluster) {
      // Weighted Bounding-Box Consensus Fusion (WBF): fuse agreeing multi-tile & full-sheet boxes when sizes are consistent
      const areaE =
        (bestCluster.xmax - bestCluster.xmin) *
        (bestCluster.ymax - bestCluster.ymin);
      const ratio =
        Math.max(areaE, areaC) / Math.max(1, Math.min(areaE, areaC));
      if (ratio <= 2.1) {
        bestCluster.members.push({ box: cand, weight });
        if (
          cand.fromTile &&
          cand.label.length > bestCluster.label.length &&
          bestCluster.label.toLowerCase() === bestCluster.category.toLowerCase()
        ) {
          bestCluster.label = cand.label;
        }
        recomputeClusterConsensus(bestCluster);
      } else {
        bestCluster.totalWeight += weight * 0.5;
      }
    } else {
      clusters.push({
        category: cand.category,
        fam: famC,
        label: cand.label,
        members: [{ box: cand, weight }],
        ymin: cand.ymin,
        xmin: cand.xmin,
        ymax: cand.ymax,
        xmax: cand.xmax,
        totalWeight: weight,
      });
    }
  }

  // Sort by consensus strength so multi-view verified symbols always rank first
  clusters.sort((a, b) => b.totalWeight - a.totalWeight);

  return clusters.map((cl) => ({
    ymin: cl.ymin,
    xmin: cl.xmin,
    ymax: cl.ymax,
    xmax: cl.xmax,
    category: cl.category,
    label: cl.label,
  }));
}

async function detectAll2DBlueprintElements(
  _ai: GoogleGenAI,
  fileName: string,
  mimeType: string,
  dataBase64: string,
  tiles?: SpatialTileInput[],
): Promise<SpatialIndex2DItem[]> {
  if (!mimeType.startsWith('image/')) return [];
  const validTiles = Array.isArray(tiles)
    ? tiles.filter(
        (t) =>
          t &&
          typeof t.dataBase64 === 'string' &&
          t.dataBase64.length > 32 &&
          t.x1 > t.x0 &&
          t.y1 > t.y0,
      ).slice(0, 6)
    : [];

  const key = hashPayloadKey(
    'idx2d-rob2',
    mimeType,
    dataBase64,
    validTiles.length,
  );
  const cached = spatialIndexCache.get(key);
  if (cached) return cached;

  const inflight = spatialIndexInFlight.get(key);
  if (inflight) return inflight;

  const task = (async (): Promise<SpatialIndex2DItem[]> => {
    try {
      const endpoint = await resolveRoboticsLabeler();
      const workerTasks: Array<() => Promise<RawCandidateBox[]>> = [];

      // 1) High-resolution overlapping tile × category specialist workers (up to 6 × 7 = 42 parallel tasks)
      for (const tile of validTiles) {
        for (const spec of SPECIALIST_SPECS) {
          if (!spec.tileAllowed) continue;
          const specialistTraining =
            trainingEngine.buildSpatialHighlightingTrainingGuide({
              category: spec.category,
            });
          workerTasks.push(async () => {
            try {
              const resp = await callRoboticsWithRetry(() =>
                endpoint.client.models.generateContent({
                  model: endpoint.model,
                  contents: {
                    parts: [
                      {
                        inlineData: {
                          mimeType: tile.mimeType || 'image/jpeg',
                          data: tile.dataBase64,
                        },
                      },
                      {
                        text: `${spec.prompt}\n\n${specialistTraining}\n\nOutput tight box_2d as [ymin, xmin, ymax, xmax] normalized 0..1000 relative to this image crop.`,
                      },
                    ],
                  },
                  config: {
                    thinkingConfig: { thinkingLevel: ThinkingLevel.HIGH },
                    responseMimeType: 'application/json',
                    responseSchema: DETECT_2D_INDEX_SCHEMA,
                  },
                }),
              );
              const parsed = parseJsonObjectSafe(resp.text ?? '{"elements":[]}');
              const rawEls = Array.isArray(parsed.elements) ? parsed.elements : [];
              const mapped: RawCandidateBox[] = [];

              for (const item of rawEls) {
                if (!item || typeof item !== 'object') continue;
                const rec = item as Record<string, unknown>;
                const norm = normalizeBoxes2D([rec]);
                if (norm.length === 0) continue;
                const lb = norm[0];

                // Discard symbols clipped right along an internal tile boundary (covered whole by overlapping neighbor tile)
                if (tile.x0 > 0 && lb.xmin <= 14) continue;
                if (tile.x1 < 1000 && lb.xmax >= 986) continue;
                if (tile.y0 > 0 && lb.ymin <= 14) continue;
                if (tile.y1 < 1000 && lb.ymax >= 986) continue;
                if (lb.xmax - lb.xmin > 850 && lb.ymax - lb.ymin > 850) continue;

                const gXMin = Math.max(
                  0,
                  Math.min(
                    1000,
                    Math.round(tile.x0 + (lb.xmin / 1000) * (tile.x1 - tile.x0)),
                  ),
                );
                const gXMax = Math.max(
                  0,
                  Math.min(
                    1000,
                    Math.round(tile.x0 + (lb.xmax / 1000) * (tile.x1 - tile.x0)),
                  ),
                );
                const gYMin = Math.max(
                  0,
                  Math.min(
                    1000,
                    Math.round(tile.y0 + (lb.ymin / 1000) * (tile.y1 - tile.y0)),
                  ),
                );
                const gYMax = Math.max(
                  0,
                  Math.min(
                    1000,
                    Math.round(tile.y0 + (lb.ymax / 1000) * (tile.y1 - tile.y0)),
                  ),
                );

                if (gXMax - gXMin < 6 || gYMax - gYMin < 6) continue;

                mapped.push({
                  ymin: gYMin,
                  xmin: gXMin,
                  ymax: gYMax,
                  xmax: gXMax,
                  category: spec.category,
                  label: String(rec.label || lb.label || spec.category).trim(),
                  fromTile: true,
                });
              }
              return mapped;
            } catch {
              return [];
            }
          });
        }
      }

      // 2) Full-sheet specialist workers (8 parallel tasks -> 42 + 8 = 50 parallel requests max)
      for (const spec of SPECIALIST_SPECS) {
        const fullSheetTraining =
          trainingEngine.buildSpatialHighlightingTrainingGuide({
            category: spec.category,
            fileName,
          });
        workerTasks.push(async () => {
          try {
            const resp = await callRoboticsWithRetry(() =>
              endpoint.client.models.generateContent({
                model: endpoint.model,
                contents: {
                  parts: [
                    { inlineData: { mimeType, data: dataBase64 } },
                    {
                      text: `Blueprint file: "${fileName}".\n${spec.prompt}\n\n${fullSheetTraining}\n\nOutput tight box_2d as [ymin, xmin, ymax, xmax] normalized 0..1000.`,
                    },
                  ],
                },
                config: {
                  thinkingConfig: { thinkingLevel: ThinkingLevel.HIGH },
                  responseMimeType: 'application/json',
                  responseSchema: DETECT_2D_INDEX_SCHEMA,
                },
              }),
            );
            const parsed = parseJsonObjectSafe(resp.text ?? '{"elements":[]}');
            const rawEls = Array.isArray(parsed.elements) ? parsed.elements : [];
            const out: RawCandidateBox[] = [];
            for (const item of rawEls) {
              if (!item || typeof item !== 'object') continue;
              const rec = item as Record<string, unknown>;
              const norm = normalizeBoxes2D([rec], undefined, spec.category === 'Space');
              if (norm.length === 0) continue;
              const b = norm[0];
              out.push({
                ymin: b.ymin,
                xmin: b.xmin,
                ymax: b.ymax,
                xmax: b.xmax,
                category: spec.category,
                label: String(rec.label || b.label || spec.category).trim(),
                fromTile: false,
              });
            }
            return out;
          } catch {
            return [];
          }
        });
      }

      const batches = await runWithConcurrencyLimit(
        workerTasks.slice(0, MAX_PARALLEL_LABEL_REQUESTS),
        MAX_PARALLEL_LABEL_REQUESTS,
      );
      const fused = fuseParallelSpatialDetections(batches.flat());

      if (fused.length > 0) {
        if (spatialIndexCache.size >= 40) {
          const oldest = spatialIndexCache.keys().next().value;
          if (oldest) spatialIndexCache.delete(oldest);
        }
        spatialIndexCache.set(key, fused);
      }
      return fused;
    } catch (err) {
      console.warn('2D Robotics spatial index warning:', err);
      return [];
    } finally {
      spatialIndexInFlight.delete(key);
    }
  })();

  spatialIndexInFlight.set(key, task);
  return task;
}

function matchFromSpatialIndex(
  index: SpatialIndex2DItem[],
  opts: {
    name: string;
    altName?: string;
    category?: string;
    isSpace?: boolean;
    expectedCount?: number;
  },
): NormalizedBox2D[] {
  if (index.length === 0) return [];
  const rawTokens = `${opts.name} ${opts.altName ?? ''}`
    .toLowerCase()
    .replace(/[^a-z0-9äöüß]+/g, ' ')
    .split(/\s+/)
    .filter((t) => t.length >= 2 && !/^\d+$/.test(t) && !['items', 'count', 'total', 'and', 'the'].includes(t));

  const singularize = (w: string) =>
    w.endsWith('ies')
      ? `${w.slice(0, -3)}y`
      : w.endsWith('es') && w.length > 4 && !w.endsWith('bles') && !w.endsWith('oves')
        ? w.slice(0, -2)
        : w.endsWith('s') && w.length > 3
          ? w.slice(0, -1)
          : w;

  const tokens = Array.from(new Set(rawTokens.map(singularize)));
  if (tokens.length === 0) return [];

  const pool = index.filter((el) => {
    const elIsSpace = el.category.toLowerCase() === 'space';
    return opts.isSpace ? elIsSpace : !elIsSpace;
  });

  // Score each indexed element against target tokens & category
  const scored = pool
    .map((el) => {
      const hay = `${el.label} ${el.category}`.toLowerCase();
      let score = 0;
      for (const tk of tokens) {
        if (hay.includes(tk)) score += 3;
      }
      if (opts.category) {
        const catLow = opts.category.toLowerCase();
        const elCat = el.category.toLowerCase();
        if (catLow === 'openings' && (elCat === 'door' || elCat === 'window')) {
          score += 1;
        } else if (
          catLow.includes('plumbing') &&
          (elCat === 'plumbing' || elCat === 'appliance')
        ) {
          score += 1;
        } else if (catLow === elCat) {
          score += 1;
        }
      }
      return { el, score };
    })
    .filter((s) => s.score >= 3)
    .sort((a, b) => b.score - a.score);

  if (scored.length === 0) return [];
  const bestScore = scored[0].score;
  const topTier = scored
    .filter((s) => s.score >= Math.max(3, bestScore - 2))
    .map((s) => ({
      ymin: s.el.ymin,
      xmin: s.el.xmin,
      ymax: s.el.ymax,
      xmax: s.el.xmax,
      label: s.el.label,
    }));

  return normalizeBoxes2D(
    topTier,
    opts.isSpace ? undefined : opts.expectedCount,
    opts.isSpace,
  );
}

const EVIDENCE_DESCRIPTION =
  'Evidence classification: "Explicit" (directly printed/in BIM), "Calculated" (derived from explicit values), "Estimated" (visually inferred), or "Unknown" (not available).';

const ANALYSIS_RESPONSE_SCHEMA = {
  type: Type.OBJECT,
  properties: {
    sceneType: {
      type: Type.STRING,
      description:
        'Classification such as "2D Floor Plan", "Interior Photo", "3D BIM Model (IFC)", "3D Mesh / Scan", or "2D CAD Drawing (DXF/PDF)".',
    },
    summary: {
      type: Type.STRING,
      description:
        'Clear 2-3 sentence overview noting verified areas, any printed-vs-calculated discrepancies, lighting fixtures, and main contents without inventing missing data.',
    },
    scaleAndCalibration: {
      type: Type.STRING,
      description:
        'How dimensions and areas were determined, explicitly separating printed labels/BIM units from unscaled visual estimates.',
    },
    dimensions: {
      type: Type.OBJECT,
      properties: {
        totalAreaSqM: {
          type: Type.NUMBER,
          nullable: true,
          description:
            'Primary reported total floor area in m² (prefer printed total if explicit, otherwise calculated sum; never overwrite conflicts silently).',
        },
        totalAreaSqFt: {
          type: Type.NUMBER,
          nullable: true,
          description: 'Primary reported total floor area in ft².',
        },
        totalAreaEvidence: {
          type: Type.STRING,
          description: EVIDENCE_DESCRIPTION,
        },
        printedLivingAreaSqM: {
          type: Type.NUMBER,
          nullable: true,
          description:
            'Exact living/interior summary area printed on the drawing in m², or null if not printed.',
        },
        printedTotalAreaSqM: {
          type: Type.NUMBER,
          nullable: true,
          description:
            'Exact overall total area printed on the drawing summary in m², or null if not printed.',
        },
        sumInteriorRoomsSqM: {
          type: Type.NUMBER,
          nullable: true,
          description:
            'Arithmetic sum of all individually labelled interior room areas in m², or null if rooms lack area values.',
        },
        sumExteriorSpacesSqM: {
          type: Type.NUMBER,
          nullable: true,
          description:
            'Arithmetic sum of labelled exterior/semi-exterior spaces (porch, balcony, terrace) in m², or null if none.',
        },
        calculatedInteriorPlusExteriorSqM: {
          type: Type.NUMBER,
          nullable: true,
          description:
            'Sum of labelled interior rooms plus labelled exterior/porch areas in m².',
        },
        grossFootprintAreaSqM: {
          type: Type.NUMBER,
          nullable: true,
          description:
            'Gross rectangular external footprint (external width × external length) in m², distinct from net room area.',
        },
        grossFootprintEvidence: {
          type: Type.STRING,
          description: EVIDENCE_DESCRIPTION,
        },
        footprintDimensionsText: {
          type: Type.STRING,
          description:
            'External overall dimensions as printed or measured, e.g. "12.00 m × 9.00 m", or "Not printed".',
        },
        estimatedCeilingHeightM: {
          type: Type.NUMBER,
          nullable: true,
          description:
            'Ceiling height in metres ONLY if explicitly printed, in a section/elevation, or from 3D geometry. Must be null on standard 2D floor plans without height annotations.',
        },
        ceilingHeightEvidence: {
          type: Type.STRING,
          description: EVIDENCE_DESCRIPTION,
        },
        estimatedVolumeCuM: {
          type: Type.NUMBER,
          nullable: true,
          description:
            'Interior volume in m³ ONLY if both reliable floor area and explicit/3D ceiling height are available. Otherwise must be null.',
        },
        volumeEvidence: {
          type: Type.STRING,
          description: EVIDENCE_DESCRIPTION,
        },
        footprintSummary: {
          type: Type.STRING,
          description:
            'Concise summary distinguishing external footprint dimensions from printed/summed floor area.',
        },
      },
      required: [
        'totalAreaEvidence',
        'grossFootprintEvidence',
        'footprintDimensionsText',
        'ceilingHeightEvidence',
        'volumeEvidence',
        'footprintSummary',
      ],
    },
    rooms: {
      type: Type.ARRAY,
      items: {
        type: Type.OBJECT,
        properties: {
          name: { type: Type.STRING },
          labelInDrawing: { type: Type.STRING },
          isExteriorOrPorch: {
            type: Type.BOOLEAN,
            description:
              'True for porches, balconies, patios, or carports; false for enclosed interior rooms.',
          },
          estimatedAreaSqM: { type: Type.NUMBER, nullable: true },
          estimatedAreaSqFt: { type: Type.NUMBER, nullable: true },
          areaEvidence: {
            type: Type.STRING,
            description: EVIDENCE_DESCRIPTION,
          },
          dimensionsText: {
            type: Type.STRING,
            description:
              'Printed wall dimensions (e.g. "4.20 m × 3.50 m"), or "Not printed" if not shown.',
          },
          dimensionsEvidence: {
            type: Type.STRING,
            description: EVIDENCE_DESCRIPTION,
          },
          notes: { type: Type.STRING },
          boxes2D: {
            type: Type.ARRAY,
            description:
              'For 2D blueprints/floorplans/photos: normalized [ymin, xmin, ymax, xmax] (0..1000) bounding box enclosing the interior wall boundary of this room/space.',
            items: BOX_2D_ITEMS_SCHEMA,
          },
        },
        required: [
          'name',
          'labelInDrawing',
          'areaEvidence',
          'dimensionsText',
          'dimensionsEvidence',
          'notes',
          'boxes2D',
        ],
      },
    },
    detectedObjects: {
      type: Type.ARRAY,
      items: {
        type: Type.OBJECT,
        properties: {
          category: {
            type: Type.STRING,
            description:
              'One of: "Lighting", "Openings", "Furniture", "Plumbing & Appliances", "Structural", "Other".',
          },
          name: { type: Type.STRING },
          count: { type: Type.INTEGER },
          unit: { type: Type.STRING },
          evidence: {
            type: Type.STRING,
            description: EVIDENCE_DESCRIPTION,
          },
          confidence: {
            type: Type.STRING,
            description: 'One of: "High", "Medium", "Estimated".',
          },
          locationsOrNotes: { type: Type.STRING },
          boxes2D: {
            type: Type.ARRAY,
            description:
              'For 2D blueprints/floorplans/photos: normalized [ymin, xmin, ymax, xmax] (0..1000) bounding boxes tightly enclosing each individual symbol/instance of this object.',
            items: BOX_2D_ITEMS_SCHEMA,
          },
        },
        required: [
          'category',
          'name',
          'count',
          'unit',
          'evidence',
          'confidence',
          'locationsOrNotes',
          'boxes2D',
        ],
      },
    },
    discrepanciesAndUncertainties: {
      type: Type.ARRAY,
      items: { type: Type.STRING },
      description:
        'Explicit list of any arithmetic discrepancies (e.g. printed summary vs room sum), footprint-vs-net-area differences, missing ceiling height/volume, and unprinted dimensions.',
    },
    datasetPatternsApplied: {
      type: Type.ARRAY,
      items: { type: Type.STRING },
      description:
        'Specific Junction 2026 dataset rules or conventions applied during this analysis.',
    },
  },
  required: [
    'sceneType',
    'summary',
    'scaleAndCalibration',
    'dimensions',
    'rooms',
    'detectedObjects',
    'discrepanciesAndUncertainties',
    'datasetPatternsApplied',
  ],
};

const ASK_RESPONSE_SCHEMA = {
  type: Type.OBJECT,
  properties: {
    answer: {
      type: Type.STRING,
      description:
        'Clean, structured plain-text answer strictly in English (NEVER raw JSON, NEVER code blocks, NEVER asterisks ** or *). State exact quantities, units, and evidence classification (Explicit/Calculated/Estimated/Unknown).',
    },
    highlights: {
      type: Type.ARRAY,
      description:
        'Object groups or rooms mentioned in the answer. Each highlight MUST strictly target ONLY the exact matching subset of elements (e.g. if answer says 6 interior doors, target ONLY those 6 doors, never all doors).',
      items: {
        type: Type.OBJECT,
        properties: {
          id: { type: Type.STRING },
          label: {
            type: Type.STRING,
            description:
              'Short human label for the highlight button, e.g. "Interior Doors (6)" or "Kitchen (11.4 m²)".',
          },
          category: {
            type: Type.STRING,
            description:
              'One of: "Lighting", "Openings", "Furniture", "Plumbing & Appliances", "Structural", "Space", "All".',
          },
          keywords: {
            type: Type.ARRAY,
            items: { type: Type.STRING },
            description:
              'Required qualifier + object keywords (all distinct concepts are AND-matched, e.g. ["interior", "door"] or ["pendant", "lamp"]).',
          },
          ifcTypes: {
            type: Type.ARRAY,
            items: { type: Type.STRING },
            description:
              'Optional IFC class names like IFCDOOR, IFCWINDOW, IFCLIGHTFIXTURE, IFCSPACE, IFCWALL.',
          },
          exactNames: {
            type: Type.ARRAY,
            items: { type: Type.STRING },
            description:
              'Optional exact IFC/mesh/room labels when matching specific types (e.g. "M_Single-Flush:0810 x 2134mm").',
          },
          ifcIds: {
            type: Type.ARRAY,
            items: { type: Type.STRING },
            description:
              'For IFC models: exact matching entity IDs from DOOR INSTANCES / WINDOW INSTANCES / LIGHTING INSTANCES / IFCSPACE (e.g. ["#412", "#419"]) so ONLY those exact elements highlight.',
          },
          storey: {
            type: Type.STRING,
            description:
              'Optional storey name filter when answer refers to a specific level/floor.',
          },
          excludeKeywords: {
            type: Type.ARRAY,
            items: { type: Type.STRING },
            description:
              'Optional keywords to exclude non-matching elements (e.g. ["exterior", "double", "switch"]).',
          },
          expectedCount: {
            type: Type.INTEGER,
            description:
              'Exact number of matching items stated in the answer (e.g. 6 if the answer says 6 doors match).',
          },
          filterDescription: {
            type: Type.STRING,
            description:
              'Concise criteria describing WHICH specific items qualify (e.g. "Single interior doors in bedrooms, excluding balcony and entrance doors").',
          },
          boxes2D: {
            type: Type.ARRAY,
            description:
              'For ALL 2D blueprints/drawings/photos: normalized [ymin, xmin, ymax, xmax] (0..1000) bounding boxes tightly enclosing ONLY the exact matching symbols or rooms stated in the answer (boxes2D.length should equal expectedCount).',
            items: BOX_2D_ITEMS_SCHEMA,
          },
        },
        required: ['id', 'label', 'keywords', 'boxes2D'],
      },
    },
  },
  required: ['answer', 'highlights'],
};

const LOCATE_2D_SCHEMA = {
  type: Type.OBJECT,
  properties: {
    boxes2D: {
      type: Type.ARRAY,
      description:
        'Exact list of normalized [ymin, xmin, ymax, xmax] (0..1000) bounding boxes tightly enclosing ONLY the matching architectural symbols or room spaces in the 2D blueprint.',
      items: BOX_2D_ITEMS_SCHEMA,
    },
  },
  required: ['boxes2D'],
};

function getAiClient(apiVersion?: string): GoogleGenAI {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error(
      'GEMINI_API_KEY is not configured. Please check the Settings > Secrets panel.',
    );
  }
  return new GoogleGenAI({
    apiKey,
    ...(apiVersion ? { apiVersion } : {}),
    httpOptions: {
      ...(apiVersion ? { apiVersion } : {}),
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

// PlanQA (Python pipeline, Junction/serve.py): /api/planqa/* is streamed through unchanged, before the JSON body
// parser, so large IFC uploads are not limited by express.json and answers can take as long as the agent needs.
const PLANQA_URL = (process.env.PLANQA_URL || 'http://127.0.0.1:8765').replace(/\/+$/, '');

function proxyPlanqa(req: Request, res: Response) {
  const target = new URL(PLANQA_URL + req.originalUrl.replace(/^\/api\/planqa/, ''));
  const upstream = http.request(
    target,
    {
      method: req.method,
      headers: {
        'content-type': req.headers['content-type'] || 'application/json',
        ...(req.headers['content-length']
          ? { 'content-length': req.headers['content-length'] }
          : {}),
      },
    },
    (up) => {
      res.status(up.statusCode || 502);
      for (const h of ['content-type', 'content-length', 'cache-control']) {
        const v = up.headers[h];
        if (v) res.setHeader(h, v);
      }
      up.pipe(res);
    },
  );
  upstream.on('error', (err) => {
    if (res.headersSent) {
      res.end();
      return;
    }
    res.status(503).json({
      error: `PlanQA service unreachable at ${PLANQA_URL} (start it with "npm run pipeline" or use docker compose): ${err.message}`,
      offline: true,
    });
  });
  req.pipe(upstream);
}

async function startServer() {
  const app = express();
  app.use('/api/planqa', proxyPlanqa);
  app.use(express.json({ limit: '50mb' }));

  app.get('/api/dataset/overview', (_req: Request, res: Response) => {
    res.json({
      datasetStats: knowledge.datasetStats,
      samples: knowledge.samples.map((s) => ({
        id: s.id,
        title: s.title,
        source: s.source,
        format: s.format,
        zipPath: s.zipPath,
        uncomp_sz: s.uncomp_sz,
        groundTruthQuestions: s.groundTruthQuestions,
      })),
    });
  });

  app.get('/api/dataset/sample/:id', async (req: Request, res: Response) => {
    try {
      const sample = knowledge.samples.find((s) => s.id === req.params.id);
      if (!sample) {
        res.status(404).json({ error: 'Sample not found in dataset catalog.' });
        return;
      }
      const buf = await fetchZipEntry(sample);
      const mimeType =
        sample.format === 'png'
          ? 'image/png'
          : sample.format === 'dxf'
            ? 'application/dxf'
            : 'application/octet-stream';
      res.json({
        sample: {
          id: sample.id,
          title: sample.title,
          source: sample.source,
          format: sample.format,
          zipPath: sample.zipPath,
          uncomp_sz: sample.uncomp_sz,
          groundTruthQuestions: sample.groundTruthQuestions,
        },
        mimeType,
        dataBase64: buf.toString('base64'),
      });
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Failed to load dataset sample.';
      res.status(500).json({ error: message });
    }
  });

  app.get('/api/training/status', (_req: Request, res: Response) => {
    res.json(trainingEngine.getStatus());
  });

  app.post('/api/training/upload', (req: Request, res: Response) => {
    try {
      const { rawText, fileName } = req.body as {
        rawText: string;
        fileName?: string;
      };
      if (!rawText) {
        res.status(400).json({ error: 'rawText training payload is required.' });
        return;
      }
      const result = trainingEngine.ingestTrainingPayload(
        rawText,
        fileName || 'training-batch.json',
      );
      res.json({
        ...result,
        status: trainingEngine.getStatus(),
      });
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Failed to parse training file.';
      res.status(400).json({ error: message });
    }
  });

  app.post('/api/training/correct', (req: Request, res: Response) => {
    try {
      const { fileName, question, wrongAnswer, correctedAnswer, ruleNote } =
        req.body as {
          fileName: string;
          question: string;
          wrongAnswer?: string;
          correctedAnswer: string;
          ruleNote?: string;
        };
      if (!question?.trim() || !correctedAnswer?.trim()) {
        res
          .status(400)
          .json({ error: 'Both question and correctedAnswer are required.' });
        return;
      }
      const entry = trainingEngine.recordCorrection({
        fileName: fileName || 'general',
        question,
        wrongAnswer,
        correctedAnswer,
        ruleNote,
      });
      res.json({ entry, status: trainingEngine.getStatus() });
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Failed to save correction.';
      res.status(500).json({ error: message });
    }
  });

  app.delete('/api/training/custom', (_req: Request, res: Response) => {
    trainingEngine.clearCustomTraining();
    res.json({ status: trainingEngine.getStatus() });
  });

  app.post('/api/analyze', async (req: Request, res: Response) => {
    try {
      const {
        fileName,
        fileKind,
        mimeType,
        dataBase64,
        tiles,
        secondaryImageBase64,
        structuredSummary,
        calibrationNote,
      } = req.body as {
        fileName: string;
        fileKind: string;
        mimeType?: string;
        dataBase64?: string;
        tiles?: SpatialTileInput[];
        secondaryImageBase64?: string;
        structuredSummary?: Record<string, unknown>;
        calibrationNote?: string;
      };

      if (!fileName) {
        res.status(400).json({ error: 'fileName is required.' });
        return;
      }

      const ai = getAiClient();
      const parts: Array<
        { inlineData: { mimeType: string; data: string } } | { text: string }
      > = [];

      if (dataBase64 && mimeType) {
        parts.push({
          inlineData: {
            mimeType,
            data: dataBase64,
          },
        });
      }

      if (secondaryImageBase64) {
        parts.push({
          inlineData: {
            mimeType: 'image/png',
            data: secondaryImageBase64,
          },
        });
      }

      const grounding = trainingEngine.retrieveGroundingContext({
        fileName,
        fileKind,
        structuredSummary,
      });

      const contextLines: string[] = [
        `Analyze this uploaded ${fileKind} file named "${fileName}".`,
        'Perform an exact architectural quantity takeoff following the Junction 2026 training conventions: distinguish Gross Rectangular Footprint from Printed Area Summaries, Sum of Labelled Interior Rooms, and Porch/Exterior Spaces. Flag any discrepancy explicitly without overwriting printed values. Classify every measurement as Explicit, Calculated, Estimated, or Unknown. Mark ceiling height and interior volume as Unknown (null) unless explicitly printed or present in 3D/section geometry. Count every lamp/lighting fixture (strictly separated from lighting switches/receptacles), door, window, sink, toilet, shower/bathtub, furniture item, and appliance. When inspecting any 2D blueprint/floorplan/image (PNG/JPG/SVG/DXF/PDF), populate "boxes2D" with tight box_2d: [ymin, xmin, ymax, xmax] coordinates (0..1000) for every room boundary and every individual detected symbol instance (e.g. each door leaf + swing arc, each window glazing band, each luminaire, each sanitary fixture) so users can highlight them accurately on the blueprint.',
      ];

      if (calibrationNote) {
        contextLines.push(
          `CALIBRATION MEASUREMENT PROVIDED BY USER: ${calibrationNote}`,
        );
      }

      const formattedSummary =
        formatStructuredSummaryForPrompt(structuredSummary);
      if (formattedSummary) {
        contextLines.push(
          `DETERMINISTIC PARSED 3D/CAD METADATA FROM FILE:\n${formattedSummary}`,
        );
      }

      if (grounding.contextBlock) {
        contextLines.push(grounding.contextBlock);
      }

      parts.push({ text: contextLines.join('\n\n') });

      const is2DImage =
        fileKind.startsWith('2d-') &&
        Boolean(dataBase64 && mimeType && mimeType.startsWith('image/'));

      const [response, spatialIndex2D] = await Promise.all([
        ai.models.generateContent({
          model: 'gemini-4-hydrogen',
          contents: { parts },
          config: {
            systemInstruction: trainingEngine.buildTrainedSystemInstruction(),
            thinkingConfig: { thinkingLevel: ThinkingLevel.HIGH },
            responseMimeType: 'application/json',
            responseSchema: ANALYSIS_RESPONSE_SCHEMA,
          },
        }),
        is2DImage && mimeType && dataBase64
          ? detectAll2DBlueprintElements(
              ai,
              fileName,
              mimeType,
              dataBase64,
              tiles,
            )
          : Promise.resolve([] as SpatialIndex2DItem[]),
      ]);

      const rawText = (response.text ?? '{}').trim();
      const parsedAnalysis = parseJsonObjectSafe(rawText);
      const verifiedAnalysis = reconcileAndVerifyAnalysis(
        parsedAnalysis,
        fileKind,
      );

      // Enrich rooms & detectedObjects with pinpoint boxes from dedicated 2D spatial detector
      if (spatialIndex2D.length > 0) {
        if (Array.isArray(verifiedAnalysis.rooms)) {
          verifiedAnalysis.rooms = verifiedAnalysis.rooms.map(
            (room: Record<string, any>) => {
              const fromIdx = matchFromSpatialIndex(spatialIndex2D, {
                name: String(room.name ?? ''),
                altName:
                  typeof room.labelInDrawing === 'string'
                    ? room.labelInDrawing
                    : undefined,
                isSpace: true,
                expectedCount: 1,
              });
              if (
                fromIdx.length > 0 &&
                (!Array.isArray(room.boxes2D) || room.boxes2D.length === 0)
              ) {
                return { ...room, boxes2D: fromIdx };
              }
              return room;
            },
          );
        }

        if (Array.isArray(verifiedAnalysis.detectedObjects)) {
          verifiedAnalysis.detectedObjects =
            verifiedAnalysis.detectedObjects.map((obj: Record<string, any>) => {
              const objCount = typeof obj.count === 'number' ? obj.count : 0;
              const fromIdx = matchFromSpatialIndex(spatialIndex2D, {
                name: String(obj.name ?? ''),
                category:
                  typeof obj.category === 'string' ? obj.category : undefined,
                isSpace: false,
                expectedCount: objCount > 0 ? objCount : undefined,
              });
              if (fromIdx.length === 0) return obj;
              const existingLen = Array.isArray(obj.boxes2D)
                ? obj.boxes2D.length
                : 0;
              const targetCnt = objCount || fromIdx.length;
              if (
                existingLen === 0 ||
                Math.abs(fromIdx.length - targetCnt) <=
                  Math.abs(existingLen - targetCnt)
              ) {
                return { ...obj, boxes2D: fromIdx };
              }
              return obj;
            });
        }
      }

      res.json({
        analysis: verifiedAnalysis,
        spatialIndex2D,
        matchedGroundTruth: grounding.matchedFileQAs,
      });
    } catch (err) {
      console.error('Analyze error:', err);
      const message =
        err instanceof Error ? err.message : 'Spatial analysis failed.';
      res.status(500).json({ error: message });
    }
  });

  app.post('/api/ask', async (req: Request, res: Response) => {
    try {
      const {
        question,
        fileName,
        fileKind,
        mimeType,
        dataBase64,
        secondaryImageBase64,
        structuredSummary,
        calibrationNote,
        currentAnalysis,
        projectContextSummary,
        history,
      } = req.body as {
        question: string;
        fileName: string;
        fileKind: string;
        mimeType?: string;
        dataBase64?: string;
        secondaryImageBase64?: string;
        structuredSummary?: Record<string, unknown>;
        calibrationNote?: string;
        currentAnalysis?: unknown;
        projectContextSummary?: string;
        history?: Array<{ role: 'user' | 'assistant'; content: string }>;
      };

      if (!question || !question.trim()) {
        res.status(400).json({ error: 'Question cannot be empty.' });
        return;
      }

      const ai = getAiClient();
      const parts: Array<
        { inlineData: { mimeType: string; data: string } } | { text: string }
      > = [];

      if (dataBase64 && mimeType) {
        parts.push({
          inlineData: { mimeType, data: dataBase64 },
        });
      }
      if (secondaryImageBase64) {
        parts.push({
          inlineData: { mimeType: 'image/png', data: secondaryImageBase64 },
        });
      }

      const grounding = trainingEngine.retrieveGroundingContext({
        fileName,
        fileKind,
        question,
        structuredSummary,
      });

      const promptSections: string[] = [
        `Active File: "${fileName}" (${fileKind})`,
      ];
      if (projectContextSummary) {
        promptSections.push(`Other Files in Project Binder:\n${projectContextSummary}`);
      }
      if (calibrationNote) {
        promptSections.push(`User Scale Calibration: ${calibrationNote}`);
      }
      const formattedSummary =
        formatStructuredSummaryForPrompt(structuredSummary);
      if (formattedSummary) {
        promptSections.push(
          `Deterministic BIM/CAD Geometry & Quantities:\n${formattedSummary}`,
        );
      }
      if (currentAnalysis) {
        promptSections.push(
          `Prior Takeoff Summary:\n${JSON.stringify(currentAnalysis).slice(0, 14000)}`,
        );
      }
      if (grounding.contextBlock) {
        promptSections.push(grounding.contextBlock);
      }
      if (history && history.length > 0) {
        const recent = history
          .slice(-6)
          .map((m) => `${m.role.toUpperCase()}: ${m.content}`)
          .join('\n');
        promptSections.push(`Recent Conversation:\n${recent}`);
      }
      promptSections.push(
        `USER QUESTION: ${question}

Respond in the required JSON schema object with keys "answer" and "highlights":
1. "answer": Plain human-readable structured text strictly in English.
   STRICT FORMATTING RULES:
   - NEVER put JSON, curly braces, or code blocks inside "answer".
   - NEVER use Markdown asterisks (**bold**, *italic*, or * bullets) or heading hashes (#).
   - Present facts concisely with labeled lines (e.g. "Interior doors: 6 (Explicit)") and compact "- " or "1. " items.
   - Use single line breaks inside sections and at most one blank line between sections. No trailing spaces.
   - Always include exact numbers, units (m², ft², count), and evidence tags (Explicit, Calculated, Estimated, or Unknown).
2. "highlights": Emit a highlight target for each object group or room mentioned in "answer".
   STRICT SUBSET ACCURACY RULE:
   - If the answer identifies a specific subset (for example "6 interior doors" or "6 doors on Floor 1" out of more total doors), set "expectedCount": 6 and "filterDescription" describing the exact requirement.
   - For IFC models: populate "ifcIds" with the exact matching entity IDs (e.g. ["#412", ...]) from the instance list so ONLY those 6 elements highlight, never extra doors. Also set "storey" / "exactNames" / "excludeKeywords" when applicable.
   - For ALL 2D blueprints, floor plans, CAD drawings, PDFs, SVGs, and images: ALWAYS populate "boxes2D" with tight box_2d: [ymin, xmin, ymax, xmax] coordinates in 0..1000 enclosing ONLY those exact matching symbols or room boundaries (for example, 6 boxes tightly around the 6 matching door leaf + swing arc symbols, never boxing non-matching doors or unrelated text).`,
      );

      parts.push({ text: promptSections.join('\n\n') });

      const response = await ai.models.generateContent({
        model: 'gemini-4-hydrogen',
        contents: { parts },
        config: {
          systemInstruction: trainingEngine.buildTrainedSystemInstruction(),
          thinkingConfig: { thinkingLevel: ThinkingLevel.HIGH },
          responseMimeType: 'application/json',
          responseSchema: ASK_RESPONSE_SCHEMA,
        },
      });

      const extractAskPayload = (
        rawInput: string,
      ): { answer: string; highlights: unknown[] } => {
        let trimmed = rawInput.trim();
        // Strip markdown fences if model wrapped JSON in ```json ... ```
        trimmed = trimmed
          .replace(/^```(?:json)?\s*/i, '')
          .replace(/\s*```$/i, '')
          .trim();

        const tryParse = (s: string): Record<string, unknown> | null => {
          try {
            const obj = JSON.parse(s);
            return obj && typeof obj === 'object' && !Array.isArray(obj)
              ? (obj as Record<string, unknown>)
              : null;
          } catch {
            return null;
          }
        };

        let obj = tryParse(trimmed);
        if (!obj) {
          const firstBrace = trimmed.indexOf('{');
          const lastBrace = trimmed.lastIndexOf('}');
          if (firstBrace !== -1 && lastBrace > firstBrace) {
            obj = tryParse(trimmed.slice(firstBrace, lastBrace + 1));
          }
        }

        // Unwrap nested JSON string if "answer" itself was accidentally JSON-encoded
        if (obj && typeof obj.answer === 'string') {
          let ansText = obj.answer.trim();
          let hl = Array.isArray(obj.highlights) ? obj.highlights : [];
          if (ansText.startsWith('{') && ansText.includes('"answer"')) {
            const inner = tryParse(
              ansText
                .replace(/^```(?:json)?\s*/i, '')
                .replace(/\s*```$/i, '')
                .trim(),
            );
            if (inner && typeof inner.answer === 'string') {
              ansText = inner.answer;
              if (hl.length === 0 && Array.isArray(inner.highlights)) {
                hl = inner.highlights;
              }
            }
          }
          return { answer: ansText, highlights: hl };
        }

        // Fallback recovery if JSON was truncated or malformed: extract "answer" string field directly
        const ansMatch = trimmed.match(/"answer"\s*:\s*"((?:\\.|[^"\\])*)"/);
        if (ansMatch) {
          let recovered = ansMatch[1];
          try {
            recovered = JSON.parse(`"${recovered}"`);
          } catch {
            recovered = recovered
              .replace(/\\n/g, '\n')
              .replace(/\\"/g, '"')
              .replace(/\\\\/g, '\\');
          }
          let hl: unknown[] = [];
          const hlMatch = trimmed.match(/"highlights"\s*:\s*(\[[\s\S]*\])/);
          if (hlMatch) {
            try {
              const parsedHl = JSON.parse(hlMatch[1]);
              if (Array.isArray(parsedHl)) hl = parsedHl;
            } catch {
              // ignore truncated highlights
            }
          }
          return { answer: recovered, highlights: hl };
        }

        // If raw output still looks like JSON braces without valid structure, strip JSON scaffolding
        if (trimmed.startsWith('{') && trimmed.includes('"')) {
          const stripped = trimmed
            .replace(/^\{|\}$/g, '')
            .replace(/"highlights"\s*:\s*\[[\s\S]*$/i, '')
            .replace(/"answer"\s*:\s*/i, '')
            .replace(/^"|"$/g, '')
            .replace(/\\n/g, '\n')
            .trim();
          return { answer: stripped || 'No response generated.', highlights: [] };
        }

        return { answer: trimmed || 'No response generated.', highlights: [] };
      };

      const rawText = response.text ?? '{}';
      const extracted = extractAskPayload(rawText);

      const cleanAnswer = String(extracted.answer ?? 'No response generated.')
        .replace(/\r\n/g, '\n')
        .replace(/(\d)\s*\*\s*(\d)/g, '$1 × $2')
        .replace(/^[ \t]*[•*]+[ \t]+/gm, '- ')
        .replace(/\*+/g, '')
        .replace(/^[ \t]*#{1,6}[ \t]+/gm, '')
        .split('\n')
        .map((line) => line.replace(/[ \t]{2,}/g, ' ').trim())
        .join('\n')
        .replace(/\n{3,}/g, '\n\n')
        .trim();

      const normalizedHighlights = Array.isArray(extracted.highlights)
        ? extracted.highlights.map((rawHl) => {
            if (!rawHl || typeof rawHl !== 'object') return rawHl;
            const rec = rawHl as Record<string, unknown>;
            const expCount =
              typeof rec.expectedCount === 'number' && rec.expectedCount > 0
                ? Math.round(rec.expectedCount)
                : undefined;
            const normBoxes = normalizeBoxes2D(rec.boxes2D, expCount);
            return {
              ...rec,
              ...(normBoxes.length > 0 ? { boxes2D: normBoxes } : { boxes2D: undefined }),
            };
          })
        : [];

      res.json({
        answer: cleanAnswer || 'No response generated.',
        highlights: normalizedHighlights,
      });
    } catch (err) {
      console.error('Ask error:', err);
      const message =
        err instanceof Error ? err.message : 'Failed to answer question.';
      res.status(500).json({ error: message });
    }
  });

  app.post('/api/index-2d', async (req: Request, res: Response) => {
    try {
      const { fileName, mimeType, dataBase64, tiles } = req.body as {
        fileName?: string;
        mimeType?: string;
        dataBase64?: string;
        tiles?: SpatialTileInput[];
      };
      if (!dataBase64 || !mimeType) {
        res.status(400).json({ error: 'Image data and mimeType are required.' });
        return;
      }
      const ai = getAiClient();
      const elements = await detectAll2DBlueprintElements(
        ai,
        fileName || 'blueprint',
        mimeType,
        dataBase64,
        tiles,
      );
      const endpoint = await resolveRoboticsLabeler();
      res.json({
        elements,
        model: endpoint.model,
      });
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Failed to index 2D blueprint.';
      res.status(500).json({ error: message });
    }
  });

  app.post('/api/locate-2d', async (req: Request, res: Response) => {
    try {
      const {
        fileName,
        label,
        category,
        keywords,
        filterDescription,
        expectedCount,
        answerContext,
        mimeType,
        dataBase64,
        tiles,
      } = req.body as {
        fileName: string;
        label: string;
        category?: string;
        keywords?: string[];
        filterDescription?: string;
        expectedCount?: number;
        answerContext?: string;
        mimeType?: string;
        dataBase64?: string;
        tiles?: SpatialTileInput[];
      };
      if (!dataBase64 || !mimeType || !label) {
        res.status(400).json({ error: 'Image data and target label are required.' });
        return;
      }

      const cacheKey = hashPayloadKey(
        'loc2d-rob2',
        label,
        category,
        (keywords ?? []).join(','),
        filterDescription,
        expectedCount,
        mimeType,
        dataBase64,
      );
      const cachedBoxes = locate2DCache.get(cacheKey);
      if (cachedBoxes) {
        res.json({ boxes2D: cachedBoxes });
        return;
      }

      const endpoint = await resolveRoboticsLabeler();
      const criteriaParts: string[] = [
        `Target: "${label}"`,
        category ? `Category: ${category}` : '',
        keywords && keywords.length > 0 ? `Keywords: ${keywords.join(', ')}` : '',
        filterDescription ? `Criteria: ${filterDescription}` : '',
        expectedCount && expectedCount > 0
          ? `Exact Qualifying Count: ${expectedCount}`
          : '',
        answerContext
          ? `Context:\n${answerContext.slice(0, 900)}`
          : '',
      ].filter(Boolean);

      const trainedSymbolGuide =
        trainingEngine.buildSpatialHighlightingTrainingGuide({
          category: category || label,
          fileName,
        });

      const promptBody = `Detect the exact 2D bounding boxes (box_2d as [ymin, xmin, ymax, xmax] normalized to 0..1000) in "${fileName}" for ONLY the visible instances matching:
${criteriaParts.join('\n')}

${trainedSymbolGuide}`;

      const validTiles =
        !filterDescription &&
        category !== 'Space' &&
        Array.isArray(tiles)
          ? tiles
              .filter(
                (t) =>
                  t &&
                  typeof t.dataBase64 === 'string' &&
                  t.dataBase64.length > 32 &&
                  t.x1 > t.x0 &&
                  t.y1 > t.y0,
              )
              .slice(0, 6)
          : [];

      let boxes: NormalizedBox2D[] = [];

      if (validTiles.length > 0) {
        // Fan out in parallel across all overlapping high-res tiles + full sheet simultaneously
        const locateTasks: Array<() => Promise<RawCandidateBox[]>> = [
          ...validTiles.map((tile) => async () => {
            try {
              const resp = await callRoboticsWithRetry(() =>
                endpoint.client.models.generateContent({
                  model: endpoint.model,
                  contents: {
                    parts: [
                      {
                        inlineData: {
                          mimeType: tile.mimeType || 'image/jpeg',
                          data: tile.dataBase64,
                        },
                      },
                      { text: promptBody },
                    ],
                  },
                  config: {
                    thinkingConfig: { thinkingLevel: ThinkingLevel.HIGH },
                    responseMimeType: 'application/json',
                    responseSchema: LOCATE_2D_SCHEMA,
                  },
                }),
              );
              const parsed = parseJsonObjectSafe(resp.text ?? '{"boxes2D":[]}');
              const localBoxes = normalizeBoxes2D(parsed.boxes2D);
              const mapped: RawCandidateBox[] = [];
              for (const lb of localBoxes) {
                if (tile.x0 > 0 && lb.xmin <= 14) continue;
                if (tile.x1 < 1000 && lb.xmax >= 986) continue;
                if (tile.y0 > 0 && lb.ymin <= 14) continue;
                if (tile.y1 < 1000 && lb.ymax >= 986) continue;
                if (lb.xmax - lb.xmin > 850 && lb.ymax - lb.ymin > 850) continue;
                mapped.push({
                  ymin: Math.max(
                    0,
                    Math.min(
                      1000,
                      Math.round(tile.y0 + (lb.ymin / 1000) * (tile.y1 - tile.y0)),
                    ),
                  ),
                  xmin: Math.max(
                    0,
                    Math.min(
                      1000,
                      Math.round(tile.x0 + (lb.xmin / 1000) * (tile.x1 - tile.x0)),
                    ),
                  ),
                  ymax: Math.max(
                    0,
                    Math.min(
                      1000,
                      Math.round(tile.y0 + (lb.ymax / 1000) * (tile.y1 - tile.y0)),
                    ),
                  ),
                  xmax: Math.max(
                    0,
                    Math.min(
                      1000,
                      Math.round(tile.x0 + (lb.xmax / 1000) * (tile.x1 - tile.x0)),
                    ),
                  ),
                  category: category || 'All',
                  label: lb.label || label,
                  fromTile: true,
                });
              }
              return mapped;
            } catch {
              return [];
            }
          }),
          async () => {
            try {
              const resp = await callRoboticsWithRetry(() =>
                endpoint.client.models.generateContent({
                  model: endpoint.model,
                  contents: {
                    parts: [
                      { inlineData: { mimeType, data: dataBase64 } },
                      { text: promptBody },
                    ],
                  },
                  config: {
                    thinkingConfig: { thinkingLevel: ThinkingLevel.HIGH },
                    responseMimeType: 'application/json',
                    responseSchema: LOCATE_2D_SCHEMA,
                  },
                }),
              );
              const parsed = parseJsonObjectSafe(resp.text ?? '{"boxes2D":[]}');
              return normalizeBoxes2D(parsed.boxes2D).map((b) => ({
                ...b,
                category: category || 'All',
                label: b.label || label,
                fromTile: false,
              }));
            } catch {
              return [];
            }
          },
        ];

        const batches = await runWithConcurrencyLimit(
          locateTasks,
          MAX_PARALLEL_LABEL_REQUESTS,
        );
        const fused = fuseParallelSpatialDetections(batches.flat());
        boxes =
          expectedCount && expectedCount > 0 && fused.length > expectedCount
            ? fused.slice(0, expectedCount)
            : fused;
      } else {
        const response = await callRoboticsWithRetry(() =>
          endpoint.client.models.generateContent({
            model: endpoint.model,
            contents: {
              parts: [
                { inlineData: { mimeType, data: dataBase64 } },
                { text: promptBody },
              ],
            },
            config: {
              thinkingConfig: { thinkingLevel: ThinkingLevel.HIGH },
              responseMimeType: 'application/json',
              responseSchema: LOCATE_2D_SCHEMA,
            },
          }),
        );
        const rawLoc = response.text ?? '{"boxes2D":[]}';
        const parsed = parseJsonObjectSafe(rawLoc);
        boxes = normalizeBoxes2D(
          parsed.boxes2D,
          expectedCount,
          category === 'Space',
        );
      }

      if (boxes.length > 0) {
        if (locate2DCache.size >= 120) {
          const oldest = locate2DCache.keys().next().value;
          if (oldest) locate2DCache.delete(oldest);
        }
        locate2DCache.set(cacheKey, boxes);
      }
      res.json({ boxes2D: boxes, model: endpoint.model });
    } catch (err) {
      console.error('Locate 2D error:', err);
      const message =
        err instanceof Error ? err.message : 'Failed to locate objects in 2D image.';
      res.status(500).json({ error: message });
    }
  });

  let cachedElevenLabsVoiceId: string | null = null;

  function getOptionalElevenLabsApiKey(): string | null {
    const key = process.env.ELEVENLABS_API_KEY?.trim();
    if (
      !key ||
      key === 'MY_ELEVENLABS_API_KEY' ||
      key === 'YOUR_ELEVENLABS_API_KEY'
    ) {
      return null;
    }
    return key;
  }

  async function resolveElevenLabsVoiceId(): Promise<string> {
    const envVoice = process.env.ELEVENLABS_VOICE_ID?.trim();
    if (envVoice && envVoice !== 'EXAVITQu4vr4xnSDxMaL') return envVoice;
    if (cachedElevenLabsVoiceId) return cachedElevenLabsVoiceId;
    return 'EXAVITQu4vr4xnSDxMaL'; // Sarah (English default)
  }

  async function fetchFirstAvailableEnglishVoiceId(
    apiKey: string,
  ): Promise<string | null> {
    try {
      const resp = await fetch('https://api.elevenlabs.io/v1/voices', {
        headers: { 'xi-api-key': apiKey },
      });
      if (!resp.ok) return null;
      const data = (await resp.json()) as {
        voices?: Array<{
          voice_id?: string;
          labels?: Record<string, string>;
        }>;
      };
      const voices = Array.isArray(data.voices) ? data.voices : [];
      const enVoice = voices.find((v) => {
        const lang = String(v.labels?.language || '').toLowerCase();
        const accent = String(v.labels?.accent || '').toLowerCase();
        return (
          lang.includes('en') ||
          accent.includes('american') ||
          accent.includes('british')
        );
      });
      const chosen = enVoice?.voice_id || voices[0]?.voice_id || null;
      if (chosen) cachedElevenLabsVoiceId = chosen;
      return chosen;
    } catch {
      return null;
    }
  }

  function pcm16ToWavBuffer(pcmBytes: Buffer, sampleRate = 24000): Buffer {
    if (
      pcmBytes.length >= 12 &&
      pcmBytes.subarray(0, 4).toString('ascii') === 'RIFF' &&
      pcmBytes.subarray(8, 12).toString('ascii') === 'WAVE'
    ) {
      return pcmBytes;
    }
    const header = Buffer.alloc(44);
    const numChannels = 1;
    const bitsPerSample = 16;
    const byteRate = (sampleRate * numChannels * bitsPerSample) / 8;
    const blockAlign = (numChannels * bitsPerSample) / 8;
    header.write('RIFF', 0);
    header.writeUInt32LE(36 + pcmBytes.length, 4);
    header.write('WAVE', 8);
    header.write('fmt ', 12);
    header.writeUInt32LE(16, 16);
    header.writeUInt16LE(1, 20);
    header.writeUInt16LE(numChannels, 22);
    header.writeUInt32LE(sampleRate, 24);
    header.writeUInt32LE(byteRate, 28);
    header.writeUInt16LE(blockAlign, 32);
    header.writeUInt16LE(bitsPerSample, 34);
    header.write('data', 36);
    header.writeUInt32LE(pcmBytes.length, 40);
    return Buffer.concat([header, pcmBytes]);
  }

  app.get('/api/voice/status', (_req: Request, res: Response) => {
    const elKey = getOptionalElevenLabsApiKey();
    res.json({
      provider: elKey ? 'elevenlabs' : 'gemini',
      configured: Boolean(elKey || process.env.GEMINI_API_KEY?.trim()),
      elevenLabsConfigured: Boolean(elKey),
      language: 'en',
    });
  });

  app.post('/api/transcribe', async (req: Request, res: Response) => {
    try {
      const { audioBase64, mimeType } = req.body as {
        audioBase64?: string;
        mimeType?: string;
      };
      if (!audioBase64 || !mimeType) {
        res.status(400).json({ error: 'Audio data and mimeType are required.' });
        return;
      }

      const cleanMime = mimeType.split(';')[0] || 'audio/webm';
      const elKey = getOptionalElevenLabsApiKey();

      // 1. Primary: ElevenLabs Scribe STT (English)
      if (elKey) {
        try {
          const ext = cleanMime.includes('mp4')
            ? 'mp4'
            : cleanMime.includes('wav')
              ? 'wav'
              : cleanMime.includes('ogg')
                ? 'ogg'
                : cleanMime.includes('mpeg') || cleanMime.includes('mp3')
                  ? 'mp3'
                  : 'webm';
          const audioBuffer = Buffer.from(audioBase64, 'base64');

          const callScribe = async (modelId: string) => {
            const form = new FormData();
            form.append('model_id', modelId);
            form.append('language_code', 'en');
            form.append(
              'file',
              new Blob([audioBuffer], { type: cleanMime }),
              `recording.${ext}`,
            );

            return fetch('https://api.elevenlabs.io/v1/speech-to-text', {
              method: 'POST',
              headers: {
                'xi-api-key': elKey,
              },
              body: form,
            });
          };

          let elResp = await callScribe('scribe_v2');
          if (!elResp.ok && (elResp.status === 400 || elResp.status === 422)) {
            elResp = await callScribe('scribe_v1');
          }

          if (elResp.ok) {
            const data = (await elResp.json()) as { text?: string };
            res.json({
              text: (data.text ?? '').trim(),
              language: 'en',
              provider: 'elevenlabs',
            });
            return;
          }
        } catch {
          // Fall through seamlessly to Gemini English transcription
        }
      }

      // 2. Automatic Fallback: Gemini English Audio Transcription (v1beta gemini-3.5-transcribe -> v1 gemini-4-hydrogen)
      const promptText =
        'Transcribe the spoken audio into plain English text only. Output ONLY the exact spoken words in English without commentary, quotes, timestamps, or extra formatting. If the audio is silent or contains no speech, output nothing.';

      try {
        const aiBeta = getAiClient('v1beta');
        const betaResp = await aiBeta.models.generateContent({
          model: 'gemini-3.5-transcribe',
          contents: {
            parts: [
              { inlineData: { mimeType: cleanMime, data: audioBase64 } },
              { text: promptText },
            ],
          },
        });
        res.json({
          text: (betaResp.text ?? '').trim(),
          language: 'en',
          provider: 'gemini-transcribe',
        });
        return;
      } catch {
        const aiDefault = getAiClient();
        const fallbackResp = await aiDefault.models.generateContent({
          model: 'gemini-4-hydrogen',
          contents: {
            parts: [
              { inlineData: { mimeType: cleanMime, data: audioBase64 } },
              { text: promptText },
            ],
          },
          config: {
            thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
          },
        });
        res.json({
          text: (fallbackResp.text ?? '').trim(),
          language: 'en',
          provider: 'gemini-audio',
        });
        return;
      }
    } catch (err) {
      console.error('Transcribe error:', err);
      const message =
        err instanceof Error ? err.message : 'Audio transcription failed.';
      res.status(500).json({ error: message });
    }
  });

  app.post('/api/tts', async (req: Request, res: Response) => {
    try {
      const { text } = req.body as { text?: string };
      const cleaned = (text ?? '').trim();
      if (!cleaned) {
        res.status(400).json({ error: 'Text is required for speech synthesis.' });
        return;
      }

      const truncated = cleaned.slice(0, 3500);
      const elKey = getOptionalElevenLabsApiKey();

      // 1. Primary: ElevenLabs Neural English TTS
      if (elKey) {
        try {
          let voiceId = await resolveElevenLabsVoiceId();

          const callElevenTts = async (vId: string, modelId: string) => {
            return fetch(
              `https://api.elevenlabs.io/v1/text-to-speech/${encodeURIComponent(vId)}?output_format=mp3_44100_128`,
              {
                method: 'POST',
                headers: {
                  'xi-api-key': elKey,
                  'Content-Type': 'application/json',
                  Accept: 'audio/mpeg',
                },
                body: JSON.stringify({
                  text: truncated,
                  model_id: modelId,
                  ...(modelId === 'eleven_flash_v2_5'
                    ? { language_code: 'en' }
                    : {}),
                  voice_settings: {
                    stability: 0.5,
                    similarity_boost: 0.75,
                  },
                }),
              },
            );
          };

          let elResp = await callElevenTts(voiceId, 'eleven_flash_v2_5');
          if (elResp.status === 404 || elResp.status === 400) {
            const discovered = await fetchFirstAvailableEnglishVoiceId(elKey);
            if (discovered && discovered !== voiceId) {
              voiceId = discovered;
              elResp = await callElevenTts(voiceId, 'eleven_flash_v2_5');
            }
          }
          if (!elResp.ok && (elResp.status === 400 || elResp.status === 422)) {
            elResp = await callElevenTts(voiceId, 'eleven_multilingual_v2');
          }

          if (elResp.ok) {
            const audioBuffer = Buffer.from(await elResp.arrayBuffer());
            if (audioBuffer.length > 0) {
              res.json({
                audioBase64: audioBuffer.toString('base64'),
                mimeType: 'audio/mpeg',
                provider: 'elevenlabs',
              });
              return;
            }
          }
        } catch {
          // Fall through seamlessly to Gemini neural TTS
        }
      }

      // 2. Automatic Fallback: Gemini Neural English TTS (v1beta gemini-3.8-flash-lite-tts)
      try {
        const aiBeta = getAiClient('v1beta');
        const ttsResp = await aiBeta.models.generateContent({
          model: 'gemini-3.8-flash-lite-tts',
          contents: [
            {
              role: 'user',
              parts: [
                {
                  text: truncated.slice(0, 2200),
                },
              ],
            },
          ],
          config: {
            responseModalities: ['AUDIO'],
            speechConfig: {
              voiceConfig: {
                prebuiltVoiceConfig: { voiceName: 'Kore' },
              },
            },
          },
        });

        const inlineAudio =
          ttsResp.candidates?.[0]?.content?.parts?.find((p) => p.inlineData?.data)
            ?.inlineData;
        if (inlineAudio?.data) {
          const rawAudioBuf = Buffer.from(inlineAudio.data, 'base64');
          const reportedMime = (inlineAudio.mimeType || '').toLowerCase();
          if (reportedMime.includes('mpeg') || reportedMime.includes('mp3')) {
            res.json({
              audioBase64: rawAudioBuf.toString('base64'),
              mimeType: 'audio/mpeg',
              provider: 'gemini-tts',
            });
            return;
          }

          const rateMatch = reportedMime.match(/rate=(\d+)/);
          const sampleRate = rateMatch ? Number(rateMatch[1]) : 24000;
          const wavBuf = pcm16ToWavBuffer(rawAudioBuf, sampleRate);
          res.json({
            audioBase64: wavBuf.toString('base64'),
            mimeType: 'audio/wav',
            provider: 'gemini-tts',
          });
          return;
        }
      } catch {
        // Fall through cleanly to client-side English speechSynthesis without HTTP 500
      }

      res.json({
        audioBase64: null,
        fallback: 'browser',
        language: 'en',
        provider: 'browser-speech',
      });
    } catch (err) {
      console.error('TTS error:', err);
      res.json({
        audioBase64: null,
        fallback: 'browser',
        language: 'en',
        provider: 'browser-speech',
      });
    }
  });

  // Always return JSON for API body-parser / payload errors (never HTML)
  app.use(
    (err: unknown, req: Request, res: Response, next: NextFunction) => {
      if (req.path.startsWith('/api/')) {
        const status =
          typeof (err as { status?: unknown })?.status === 'number'
            ? ((err as { status: number }).status)
            : 500;
        const msg =
          status === 413
            ? 'Uploaded file payload exceeds the 50 MB request limit.'
            : err instanceof Error
              ? err.message
              : 'Unexpected server error.';
        res.status(status).json({ error: msg });
        return;
      }
      next(err);
    },
  );

  const httpServer = http.createServer(app);
  const isProd = process.env.NODE_ENV === 'production';
  if (!isProd) {
    const vite = await createViteServer({
      server: {
        middlewareMode: true,
        hmr: false,
        watch: null,
      },
      appType: 'spa',
    });

    app.get('/@vite/client', async (_req: Request, res: Response, next) => {
      try {
        const result = await vite.transformRequest('/@vite/client');
        if (result?.code) {
          const patched = result.code.replace(
            'transport.connect(createHMRHandler(handleMessage));',
            '/* HMR WebSocket disabled in middleware mode */',
          );
          res.type('application/javascript').send(patched);
          return;
        }
      } catch {
        // Fall through to Vite middleware
      }
      next();
    });

    app.use(vite.middlewares);
  } else {
    const distDir = path.join(__dirname, 'dist');
    app.use(express.static(distDir));
    app.get('*', (_req: Request, res: Response) => {
      res.sendFile(path.join(distDir, 'index.html'));
    });
  }

  const port = Number(process.env.PORT || 3000);
  httpServer.listen(port, '0.0.0.0', () => {
    console.log(`Stoa server running on http://0.0.0.0:${port}`);
  });
}

startServer().catch((err) => {
  console.error('Failed to start server:', err);
  process.exit(1);
});
