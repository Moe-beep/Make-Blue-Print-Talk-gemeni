export type FileKind =
  | '2d-image'
  | '2d-svg'
  | '2d-pdf'
  | '2d-dxf'
  | '3d-ifc'
  | '3d-mesh';

export interface GroundTruthQA {
  question: string;
  answer: string | number;
  unit: string | null;
  type: string;
}

export interface DatasetSampleMeta {
  id: string;
  title: string;
  source: string;
  format: 'png' | 'dxf' | 'ifc';
  zipPath: string;
  uncomp_sz: number;
  groundTruthQuestions: GroundTruthQA[];
}

export interface DatasetOverview {
  datasetStats: {
    totalFilesInZip: number;
    totalQuestions: number;
    uniqueAnnotatedDrawings: number;
    questionTypes: Record<string, number>;
    sourceDistribution: Record<string, number>;
  };
  samples: DatasetSampleMeta[];
}

export interface RoomMetric {
  name: string;
  labelInDrawing: string;
  estimatedAreaSqM: number | null;
  estimatedAreaSqFt: number | null;
  dimensionsText: string;
  notes: string;
}

export interface DetectedObject {
  category:
    | 'Lighting'
    | 'Openings'
    | 'Furniture'
    | 'Plumbing & Appliances'
    | 'Structural'
    | 'Other';
  name: string;
  count: number;
  unit: string;
  confidence: 'High' | 'Medium' | 'Estimated';
  locationsOrNotes: string;
}

export interface TakeoffAnalysis {
  sceneType: string;
  summary: string;
  scaleAndCalibration: string;
  dimensions: {
    totalAreaSqM: number | null;
    totalAreaSqFt: number | null;
    estimatedCeilingHeightM: number | null;
    estimatedVolumeCuM: number | null;
    footprintSummary: string;
  };
  rooms: RoomMetric[];
  detectedObjects: DetectedObject[];
  datasetPatternsApplied: string[];
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
}

export interface ParsedIfcSummary {
  schema: string;
  authoringTool: string;
  projectName: string;
  lengthUnit: string;
  lengthToMeters: number;
  storeys: Array<{ name: string; elevationM: number }>;
  spaces: Array<{
    id: string;
    code: string;
    name: string;
    areaSqM: number | null;
    volumeCuM: number | null;
  }>;
  elementCounts: Record<string, number>;
  lightingFixtures: Array<{ name: string; count: number }>;
  furnitureAndFixtures: Array<{ name: string; count: number }>;
  boundingBoxMeters: { width: number; depth: number; height: number } | null;
  totalMeshCount: number;
}

export interface ParsedDxfSummary {
  units: string;
  layers: Array<{ name: string; entityCount: number }>;
  entityCounts: Record<string, number>;
  blocks: Array<{ name: string; count: number }>;
  texts: string[];
  boundingBox: {
    minX: number;
    minY: number;
    maxX: number;
    maxY: number;
    width: number;
    height: number;
  } | null;
}

export interface LoadedAsset {
  id: string;
  name: string;
  fileKind: FileKind;
  mimeType: string;
  sizeBytes: number;
  previewUrl?: string;
  rawText?: string;
  arrayBuffer?: ArrayBuffer;
  sampleMeta?: DatasetSampleMeta;
  structuredSummary?: Record<string, unknown>;
}
