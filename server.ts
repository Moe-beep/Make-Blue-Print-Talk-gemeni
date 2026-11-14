import 'dotenv/config';
import express from 'express';
import type { Request, Response } from 'express';
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
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

function reconcileAndVerifyAnalysis(
  rawAnalysis: Record<string, unknown>,
  fileKind: string,
): Record<string, unknown> {
  const analysis = structuredClone(rawAnalysis) as {
    dimensions?: {
      totalAreaSqM: number | null;
      totalAreaSqFt: number | null;
      totalAreaEvidence: string;
      printedLivingAreaSqM: number | null;
      printedTotalAreaSqM: number | null;
      sumInteriorRoomsSqM: number | null;
      sumExteriorSpacesSqM: number | null;
      calculatedInteriorPlusExteriorSqM: number | null;
      grossFootprintAreaSqM: number | null;
      grossFootprintEvidence: string;
      footprintDimensionsText: string;
      estimatedCeilingHeightM: number | null;
      ceilingHeightEvidence: string;
      estimatedVolumeCuM: number | null;
      volumeEvidence: string;
      footprintSummary: string;
    };
    rooms?: Array<{
      name: string;
      labelInDrawing: string;
      isExteriorOrPorch?: boolean;
      estimatedAreaSqM: number | null;
      estimatedAreaSqFt: number | null;
      areaEvidence: string;
      dimensionsText: string;
      dimensionsEvidence: string;
      notes: string;
    }>;
    discrepanciesAndUncertainties?: string[];
  };

  const rooms = Array.isArray(analysis.rooms) ? analysis.rooms : [];
  const dims = analysis.dimensions ?? ({} as NonNullable<typeof analysis.dimensions>);
  const discrepancies = new Set<string>(
    Array.isArray(analysis.discrepanciesAndUncertainties)
      ? analysis.discrepanciesAndUncertainties
      : [],
  );

  // Sync m² <-> ft² per room
  let interiorSum = 0;
  let hasInterior = false;
  let exteriorSum = 0;
  let hasExterior = false;

  for (const r of rooms) {
    if (
      typeof r.estimatedAreaSqM === 'number' &&
      Number.isFinite(r.estimatedAreaSqM)
    ) {
      r.estimatedAreaSqM = round2(r.estimatedAreaSqM);
      r.estimatedAreaSqFt = round2(r.estimatedAreaSqM * SQM_TO_SQFT);
    } else if (
      typeof r.estimatedAreaSqFt === 'number' &&
      Number.isFinite(r.estimatedAreaSqFt)
    ) {
      r.estimatedAreaSqM = round2(r.estimatedAreaSqFt / SQM_TO_SQFT);
      r.estimatedAreaSqFt = round2(r.estimatedAreaSqFt);
    }

    if (typeof r.estimatedAreaSqM === 'number') {
      if (r.isExteriorOrPorch) {
        exteriorSum += r.estimatedAreaSqM;
        hasExterior = true;
      } else {
        interiorSum += r.estimatedAreaSqM;
        hasInterior = true;
      }
    }
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
    typeof dims.printedLivingAreaSqM === 'number' &&
    typeof dims.sumInteriorRoomsSqM === 'number'
  ) {
    const diff = Math.abs(dims.printedLivingAreaSqM - dims.sumInteriorRoomsSqM);
    if (diff > 0.05) {
      discrepancies.add(
        `Arithmetic discrepancy of ${diff.toFixed(2)} m² between printed living area summary (${dims.printedLivingAreaSqM.toFixed(2)} m²) and sum of labelled interior rooms (${dims.sumInteriorRoomsSqM.toFixed(2)} m²). Both values preserved.`,
      );
    }
  }
  if (
    typeof dims.printedTotalAreaSqM === 'number' &&
    typeof dims.calculatedInteriorPlusExteriorSqM === 'number'
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
  }

  // Ensure totalAreaSqFt matches totalAreaSqM
  if (
    typeof dims.totalAreaSqM === 'number' &&
    Number.isFinite(dims.totalAreaSqM)
  ) {
    dims.totalAreaSqM = round2(dims.totalAreaSqM);
    dims.totalAreaSqFt = round2(dims.totalAreaSqM * SQM_TO_SQFT);
  } else if (
    typeof dims.totalAreaSqFt === 'number' &&
    Number.isFinite(dims.totalAreaSqFt)
  ) {
    dims.totalAreaSqM = round2(dims.totalAreaSqFt / SQM_TO_SQFT);
    dims.totalAreaSqFt = round2(dims.totalAreaSqFt);
  }

  analysis.dimensions = dims;
  analysis.rooms = rooms;
  analysis.discrepanciesAndUncertainties = Array.from(discrepancies);
  return analysis as Record<string, unknown>;
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
              'For 2D floorplan images/photos only: normalized 0..1000 bounding boxes locating this space in the image.',
            items: {
              type: Type.OBJECT,
              properties: {
                ymin: { type: Type.INTEGER },
                xmin: { type: Type.INTEGER },
                ymax: { type: Type.INTEGER },
                xmax: { type: Type.INTEGER },
                label: { type: Type.STRING },
              },
              required: ['ymin', 'xmin', 'ymax', 'xmax'],
            },
          },
        },
        required: [
          'name',
          'labelInDrawing',
          'areaEvidence',
          'dimensionsText',
          'dimensionsEvidence',
          'notes',
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
              'For 2D images/photos only: normalized 0..1000 bounding boxes locating visible instances of this object in the image.',
            items: {
              type: Type.OBJECT,
              properties: {
                ymin: { type: Type.INTEGER },
                xmin: { type: Type.INTEGER },
                ymax: { type: Type.INTEGER },
                xmax: { type: Type.INTEGER },
                label: { type: Type.STRING },
              },
              required: ['ymin', 'xmin', 'ymax', 'xmax'],
            },
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
        'Clean, structured plain-text answer in the user language (NEVER raw JSON, NEVER code blocks, NEVER asterisks ** or *). State exact quantities, units, and evidence classification (Explicit/Calculated/Estimated/Unknown).',
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
              'For 2D floorplan images/photos: normalized 0..1000 bounding boxes locating ONLY the exact instances meeting the criteria (count in boxes2D should equal expectedCount).',
            items: {
              type: Type.OBJECT,
              properties: {
                ymin: { type: Type.INTEGER },
                xmin: { type: Type.INTEGER },
                ymax: { type: Type.INTEGER },
                xmax: { type: Type.INTEGER },
                label: { type: Type.STRING },
              },
              required: ['ymin', 'xmin', 'ymax', 'xmax'],
            },
          },
        },
        required: ['id', 'label', 'keywords'],
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
      items: {
        type: Type.OBJECT,
        properties: {
          ymin: { type: Type.INTEGER },
          xmin: { type: Type.INTEGER },
          ymax: { type: Type.INTEGER },
          xmax: { type: Type.INTEGER },
          label: { type: Type.STRING },
        },
        required: ['ymin', 'xmin', 'ymax', 'xmax'],
      },
    },
  },
  required: ['boxes2D'],
};

function getAiClient(): GoogleGenAI {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error(
      'GEMINI_API_KEY is not configured. Please check the Settings > Secrets panel.',
    );
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

async function startServer() {
  const app = express();
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
        res.status(400).json({ error: 'rawText of questions JSON/JSONL is required.' });
        return;
      }
      const result = trainingEngine.ingestTrainingPayload(
        rawText,
        fileName || 'questions.json',
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
        secondaryImageBase64,
        structuredSummary,
        calibrationNote,
      } = req.body as {
        fileName: string;
        fileKind: string;
        mimeType?: string;
        dataBase64?: string;
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
        'Perform an exact architectural quantity takeoff following the Junction 2026 training conventions: distinguish Gross Rectangular Footprint from Printed Area Summaries, Sum of Labelled Interior Rooms, and Porch/Exterior Spaces. Flag any discrepancy explicitly without overwriting printed values. Classify every measurement as Explicit, Calculated, Estimated, or Unknown. Mark ceiling height and interior volume as Unknown (null) unless explicitly printed or present in 3D/section geometry. Count every lamp/lighting fixture (strictly separated from lighting switches/receptacles), door, window, sink, toilet, shower/bathtub, furniture item, and appliance. When inspecting a 2D image/floorplan, also populate "boxes2D" with normalized [0..1000] bounding boxes for each room and detected object instance so users can highlight them in the drawing.',
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

      const response = await ai.models.generateContent({
        model: 'gemini-4-hydrogen',
        contents: { parts },
        config: {
          systemInstruction: trainingEngine.buildTrainedSystemInstruction(),
          thinkingConfig: { thinkingLevel: ThinkingLevel.HIGH },
          responseMimeType: 'application/json',
          responseSchema: ANALYSIS_RESPONSE_SCHEMA,
        },
      });

      const rawText = (response.text ?? '{}').trim();
      const parsedAnalysis = JSON.parse(rawText) as Record<string, unknown>;
      const verifiedAnalysis = reconcileAndVerifyAnalysis(
        parsedAnalysis,
        fileKind,
      );
      res.json({
        analysis: verifiedAnalysis,
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
1. "answer": Plain human-readable structured text in the same language as the user's question.
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
   - For 2D raster images/floorplans: populate "boxes2D" with bounding boxes for ONLY those exact matching instances (e.g. 6 boxes for 6 doors, never boxing non-matching doors).`,
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

      res.json({
        answer: cleanAnswer || 'No response generated.',
        highlights: extracted.highlights,
      });
    } catch (err) {
      console.error('Ask error:', err);
      const message =
        err instanceof Error ? err.message : 'Failed to answer question.';
      res.status(500).json({ error: message });
    }
  });

  app.post('/api/locate-2d', async (req: Request, res: Response) => {
    try {
      const {
        fileName,
        label,
        keywords,
        filterDescription,
        expectedCount,
        answerContext,
        mimeType,
        dataBase64,
      } = req.body as {
        fileName: string;
        label: string;
        keywords?: string[];
        filterDescription?: string;
        expectedCount?: number;
        answerContext?: string;
        mimeType?: string;
        dataBase64?: string;
      };
      if (!dataBase64 || !mimeType || !label) {
        res.status(400).json({ error: 'Image data and target label are required.' });
        return;
      }
      const ai = getAiClient();
      const criteriaParts: string[] = [
        `Target: "${label}"`,
        keywords && keywords.length > 0 ? `Keywords: ${keywords.join(', ')}` : '',
        filterDescription ? `Exact Selection Criteria: ${filterDescription}` : '',
        expectedCount ? `Expected Exact Matching Count: ${expectedCount}` : '',
        answerContext
          ? `Verified Answer Context:\n${answerContext.slice(0, 1200)}`
          : '',
      ].filter(Boolean);

      const response = await ai.models.generateContent({
        model: 'gemini-4-hydrogen',
        contents: {
          parts: [
            { inlineData: { mimeType, data: dataBase64 } },
            {
              text: `Locate ONLY the specific visible instances in "${fileName}" that satisfy ALL of these requirements:
${criteriaParts.join('\n')}

STRICT RULE:
- Do NOT include bounding boxes for elements outside these exact requirements (for example, if only 6 interior doors match the criteria out of 10 total doors, return ONLY those 6 doors and exclude the other 4).
- Return normalized [0..1000] ymin, xmin, ymax, xmax bounding boxes in JSON.`,
            },
          ],
        },
        config: {
          responseMimeType: 'application/json',
          responseSchema: LOCATE_2D_SCHEMA,
        },
      });
      const rawLoc = (response.text ?? '{"boxes2D":[]}')
        .trim()
        .replace(/^```(?:json)?\s*/i, '')
        .replace(/\s*```$/i, '')
        .trim();
      const parsed = JSON.parse(rawLoc);
      let boxes = Array.isArray(parsed.boxes2D) ? parsed.boxes2D : [];
      if (expectedCount && expectedCount > 0 && boxes.length > expectedCount) {
        boxes = boxes.slice(0, expectedCount);
      }
      res.json({ boxes2D: boxes });
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Failed to locate objects in 2D image.';
      res.status(500).json({ error: message });
    }
  });

  const httpServer = http.createServer(app);
  const isProd = process.env.NODE_ENV === 'production';
  if (!isProd) {
    const disableHmr = process.env.DISABLE_HMR === 'true';
    const vite = await createViteServer({
      server: {
        middlewareMode: true,
        ...(disableHmr ? {} : { ws: { server: httpServer } }),
      },
      appType: 'spa',
    });

    if (disableHmr) {
      app.get('/@vite/client', async (_req: Request, res: Response, next) => {
        try {
          const result = await vite.transformRequest('/@vite/client');
          if (result?.code) {
            const patched = result.code.replace(
              'transport.connect(createHMRHandler(handleMessage));',
              '/* HMR disabled via DISABLE_HMR */',
            );
            res.type('application/javascript').send(patched);
            return;
          }
        } catch {
          // Fall through to Vite middleware
        }
        next();
      });
    }

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
