import 'dotenv/config';
import express from 'express';
import type { Request, Response } from 'express';
import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import { fileURLToPath } from 'node:url';
import { GoogleGenAI, Type, ThinkingLevel } from '@google/genai';
import { createServer as createViteServer } from 'vite';

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

function buildSystemInstruction(): string {
  const exemplarLines: string[] = [];
  for (const [bucket, items] of Object.entries(knowledge.exemplars)) {
    for (const item of items.slice(0, 2)) {
      exemplarLines.push(
        `- [${bucket}] Q: "${item.question}" -> A: ${JSON.stringify(item.answer)} ${item.unit ?? ''}`,
      );
    }
  }

  return `You are Stoa AI, a specialized architectural, interior, and BIM/CAD quantity-takeoff spatial analyst trained on the Junction 2026 construction drawings dataset (${knowledge.datasetStats.totalFilesInZip.toLocaleString()} graphics files across 2D floor plans, CAD DXFs, PDFs, and 3D IFC models, with ${knowledge.datasetStats.totalQuestions.toLocaleString()} verified quantity-takeoff questions).

CORE LEARNED CONVENTIONS FROM THE JUNCTION 2026 DATASET:
1. Multilingual Floor Plan Labels:
   - Finnish (CubiCasa5k / Finna): MH = Makuuhuone (Bedroom), OH = Olohuone (Living Room), K / KT = Keittiö / Keittotila (Kitchen), PH / KPH = Pesuhuone / Kylpyhuone (Bathroom), S = Sauna, WC = Toilet, ET = Eteinen (Entry/Hallway), VH = Vaatehuone (Walk-in Closet), KHH = Kodinhoitohuone (Utility Room), PARV / P = Parveke (Balcony).
   - German/French/Italian (Swiss Dwellings): Schlafzimmer / Chambre / Camera = Bedroom, Wohnzimmer / Séjour / Soggiorno = Living Room, Küche / Cuisine / Cucina = Kitchen, Bad / WC = Bathroom, Korridor / Entrée = Hallway, Balkon / Balcon = Balcony.
2. Door & Window Counting Conventions:
   - Count doorways fitted with a door leaf/swing arc; open archways or cased openings without a door leaf do NOT count as doors unless explicitly asked.
   - Double doors count as 1 double-door unit (or note leaf count clearly).
   - Count distinct window openings/frames along exterior/interior walls.
3. Lighting & Lamps Detection (CRITICAL USER PRIORITY):
   - Carefully inspect photos, floor plans, CAD blocks, and 3D/IFC models for ALL lighting fixtures and lamps: ceiling pendants, chandeliers, recessed downlights/spotlights, wall sconces, floor lamps, desk/table lamps, fluorescent/linear light strips, and IFC \`IfcLightFixture\` / \`IfcFlowTerminal\` lighting families.
   - Distinguish natural light sources (windows/skylights) from artificial lamps/luminaires, and list each lamp type and exact count clearly.
4. Room Size & Spatial Area Estimation:
   - If deterministic IFC \`IFCSPACE\` / \`IFCELEMENTQUANTITY\` areas, DXF bounding boxes, explicit printed room m²/ft² labels, scale bars, or user-calibrated reference measurements are provided, use those exact figures.
   - For unscaled interior photos or unannotated PNG floor plans, calibrate scale carefully using architectural priors (standard interior door width ~0.85–0.90 m, standard bed ~1.6×2.0 m or 0.9×2.0 m, kitchen counter depth ~0.60 m, toilet bowl ~0.40×0.65 m, standard ceiling height ~2.4–2.7 m) and clearly state whether dimensions are exact (from BIM/CAD/labels/calibration) or visually estimated.
5. Complete Object Inventory:
   - Count every visible or modeled object group: lamps/lights, doors, windows, seating (sofas, armchairs, dining chairs, stools), tables/desks, beds, cabinets/wardrobes, plumbing/appliances (toilets/WC, sinks/basins, showers, bathtubs, stoves, fridges, saunas), and structural elements when relevant.

SAMPLE GROUND-TRUTH PATTERNS LEARNED FROM JUNCTION 2026:
${exemplarLines.join('\n')}`;
}

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
        'Clear 2-3 sentence overview of the space, room layout, dimensions, lighting fixtures, and main contents.',
    },
    scaleAndCalibration: {
      type: Type.STRING,
      description:
        'How dimensions and areas were determined (e.g., exact IFC quantities, CAD units, printed labels, user-calibrated ruler, or architectural fixture scale priors).',
    },
    dimensions: {
      type: Type.OBJECT,
      properties: {
        totalAreaSqM: {
          type: Type.NUMBER,
          nullable: true,
          description: 'Total floor area in square metres.',
        },
        totalAreaSqFt: {
          type: Type.NUMBER,
          nullable: true,
          description: 'Total floor area in square feet.',
        },
        estimatedCeilingHeightM: {
          type: Type.NUMBER,
          nullable: true,
          description: 'Estimated or exact ceiling height per floor in metres.',
        },
        estimatedVolumeCuM: {
          type: Type.NUMBER,
          nullable: true,
          description: 'Estimated or exact interior volume in cubic metres.',
        },
        footprintSummary: {
          type: Type.STRING,
          description:
            'Human-readable overall footprint or bounding dimensions, e.g. "9.8 m × 7.4 m (2 storeys)".',
        },
      },
      required: ['footprintSummary'],
    },
    rooms: {
      type: Type.ARRAY,
      items: {
        type: Type.OBJECT,
        properties: {
          name: { type: Type.STRING },
          labelInDrawing: { type: Type.STRING },
          estimatedAreaSqM: { type: Type.NUMBER, nullable: true },
          estimatedAreaSqFt: { type: Type.NUMBER, nullable: true },
          dimensionsText: { type: Type.STRING },
          notes: { type: Type.STRING },
        },
        required: ['name', 'labelInDrawing', 'dimensionsText', 'notes'],
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
          confidence: {
            type: Type.STRING,
            description: 'One of: "High", "Medium", "Estimated".',
          },
          locationsOrNotes: { type: Type.STRING },
        },
        required: [
          'category',
          'name',
          'count',
          'unit',
          'confidence',
          'locationsOrNotes',
        ],
      },
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
    'datasetPatternsApplied',
  ],
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

      const contextLines: string[] = [
        `Analyze this uploaded ${fileKind} file named "${fileName}".`,
        'Provide a comprehensive spatial & quantity-takeoff report: room sizes (m² and ft²), ceiling height/volume estimates, EVERY lamp/lighting fixture you can detect, doors, windows, furniture, appliances, and structural items.',
      ];

      if (calibrationNote) {
        contextLines.push(`CALIBRATION MEASUREMENT PROVIDED BY USER: ${calibrationNote}`);
      }

      if (structuredSummary && Object.keys(structuredSummary).length > 0) {
        contextLines.push(
          `DETERMINISTIC PARSED 3D/CAD METADATA FROM FILE:\n${JSON.stringify(
            structuredSummary,
            null,
            2,
          ).slice(0, 28000)}`,
        );
      }

      parts.push({ text: contextLines.join('\n\n') });

      const response = await ai.models.generateContent({
        model: 'gemini-4-hydrogen',
        contents: { parts },
        config: {
          systemInstruction: buildSystemInstruction(),
          thinkingConfig: { thinkingLevel: ThinkingLevel.MEDIUM },
          responseMimeType: 'application/json',
          responseSchema: ANALYSIS_RESPONSE_SCHEMA,
        },
      });

      const rawText = (response.text ?? '{}').trim();
      const analysis = JSON.parse(rawText);
      res.json({ analysis });
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

      const promptSections: string[] = [
        `Active File: "${fileName}" (${fileKind})`,
      ];
      if (calibrationNote) {
        promptSections.push(`User Scale Calibration: ${calibrationNote}`);
      }
      if (structuredSummary && Object.keys(structuredSummary).length > 0) {
        promptSections.push(
          `Deterministic BIM/CAD Geometry & Quantities:\n${JSON.stringify(
            structuredSummary,
          ).slice(0, 20000)}`,
        );
      }
      if (currentAnalysis) {
        promptSections.push(
          `Prior Takeoff Summary:\n${JSON.stringify(currentAnalysis).slice(0, 12000)}`,
        );
      }
      if (history && history.length > 0) {
        const recent = history
          .slice(-6)
          .map((m) => `${m.role.toUpperCase()}: ${m.content}`)
          .join('\n');
        promptSections.push(`Recent Conversation:\n${recent}`);
      }
      promptSections.push(
        `USER QUESTION: ${question}\nAnswer directly, precisely, and concisely with exact numbers, units (m², ft², counts), and clear reasoning based on what is visible or parsed in the file.`,
      );

      parts.push({ text: promptSections.join('\n\n') });

      const response = await ai.models.generateContent({
        model: 'gemini-4-hydrogen',
        contents: { parts },
        config: {
          systemInstruction: buildSystemInstruction(),
          thinkingConfig: { thinkingLevel: ThinkingLevel.MEDIUM },
        },
      });

      res.json({ answer: response.text ?? 'No response generated.' });
    } catch (err) {
      console.error('Ask error:', err);
      const message =
        err instanceof Error ? err.message : 'Failed to answer question.';
      res.status(500).json({ error: message });
    }
  });

  const isProd = process.env.NODE_ENV === 'production';
  if (!isProd) {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
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
  app.listen(port, '0.0.0.0', () => {
    console.log(`Stoa server running on http://0.0.0.0:${port}`);
  });
}

startServer().catch((err) => {
  console.error('Failed to start server:', err);
  process.exit(1);
});
