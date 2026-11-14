export type FileKind =
  | '2d-image'
  | '2d-svg'
  | '2d-pdf'
  | '2d-dxf'
  | '3d-ifc'
  | '3d-mesh';

export type EvidenceClass =
  | 'Explicit'
  | 'Calculated'
  | 'Estimated'
  | 'Unknown';

export interface Box2D {
  ymin: number; // 0..1000 normalized top
  xmin: number; // 0..1000 normalized left
  ymax: number; // 0..1000 normalized bottom
  xmax: number; // 0..1000 normalized right
  label?: string;
}

export interface HighlightTarget {
  id: string;
  label: string;
  category?:
    | 'Lighting'
    | 'Openings'
    | 'Furniture'
    | 'Plumbing & Appliances'
    | 'Structural'
    | 'Space'
    | 'All';
  keywords: string[];
  ifcTypes?: string[];
  exactNames?: string[];
  ifcIds?: string[];
  storey?: string;
  excludeKeywords?: string[];
  expectedCount?: number;
  filterDescription?: string;
  boxes2D?: Box2D[];
  subTargets?: HighlightTarget[];
}

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
  isExteriorOrPorch?: boolean;
  estimatedAreaSqM: number | null;
  estimatedAreaSqFt: number | null;
  areaEvidence: EvidenceClass;
  dimensionsText: string;
  dimensionsEvidence: EvidenceClass;
  notes: string;
  boxes2D?: Box2D[];
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
  evidence: EvidenceClass;
  confidence: 'High' | 'Medium' | 'Estimated';
  locationsOrNotes: string;
  boxes2D?: Box2D[];
}

export interface TakeoffAnalysis {
  sceneType: string;
  summary: string;
  scaleAndCalibration: string;
  dimensions: {
    totalAreaSqM: number | null;
    totalAreaSqFt: number | null;
    totalAreaEvidence: EvidenceClass;
    printedLivingAreaSqM: number | null;
    printedTotalAreaSqM: number | null;
    sumInteriorRoomsSqM: number | null;
    sumExteriorSpacesSqM: number | null;
    calculatedInteriorPlusExteriorSqM: number | null;
    grossFootprintAreaSqM: number | null;
    grossFootprintEvidence: EvidenceClass;
    footprintDimensionsText: string;
    estimatedCeilingHeightM: number | null;
    ceilingHeightEvidence: EvidenceClass;
    estimatedVolumeCuM: number | null;
    volumeEvidence: EvidenceClass;
    footprintSummary: string;
  };
  rooms: RoomMetric[];
  detectedObjects: DetectedObject[];
  discrepanciesAndUncertainties: string[];
  datasetPatternsApplied: string[];
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  highlights?: HighlightTarget[];
}

export interface ParsedIfcSpace {
  id: string;
  code: string;
  name: string;
  storey: string | null;
  areaSqM: number | null;
  areaSource: 'Explicit BIM Property' | 'Calculated Extrusion' | 'Unknown';
  volumeCuM: number | null;
  heightM: number | null;
}

export interface ParsedIfcStoreyBreakdown {
  name: string;
  elevationM: number;
  spaceCount: number;
  netFloorAreaSqM: number | null;
  doorCount: number;
  windowCount: number;
  wallCount: number;
  stairCount: number;
  lightFixtureCount: number;
  sanitaryCount: number;
}

export interface ParsedIfcSummary {
  schema: string;
  authoringTool: string;
  projectName: string;
  lengthUnit: string;
  lengthToMeters: number;
  storeys: ParsedIfcStoreyBreakdown[];
  spaces: ParsedIfcSpace[];
  totalSpaceAreaSqM: number | null;
  elementCounts: Record<string, number>;
  doorTypes: Array<{ name: string; count: number; widthM?: number; heightM?: number }>;
  windowTypes: Array<{ name: string; count: number; widthM?: number; heightM?: number }>;
  doorInstances?: Array<{
    id: string;
    name: string;
    type: string;
    storey: string | null;
    widthM?: number;
    heightM?: number;
    spaces?: string[];
  }>;
  windowInstances?: Array<{
    id: string;
    name: string;
    type: string;
    storey: string | null;
    widthM?: number;
    heightM?: number;
    spaces?: string[];
  }>;
  lightingInstances?: Array<{
    id: string;
    name: string;
    type: string;
    storey: string | null;
  }>;
  lightingFixtures: Array<{ name: string; count: number }>;
  lightingControls: Array<{ name: string; count: number }>;
  electricalDevices: Array<{ name: string; count: number }>;
  sanitaryAndAppliances: Array<{ name: string; count: number }>;
  furnitureAndFixtures: Array<{ name: string; count: number }>;
  wallStats: {
    count: number;
    totalLengthM: number | null;
    totalNetSideAreaSqM: number | null;
    totalNetVolumeCuM: number | null;
  };
  slabStats: {
    count: number;
    totalNetAreaSqM: number | null;
    totalNetVolumeCuM: number | null;
  };
  boundingBoxMeters: { width: number; depth: number; height: number } | null;
  totalMeshCount: number;
}

export interface ParsedDxfRoom {
  code: string;
  label: string;
  apartment: string | null;
  areaSqM: number;
  isExterior: boolean;
}

export interface ParsedDxfApartment {
  name: string;
  indoorAreaSqM: number;
  exteriorAreaSqM: number;
  roomCount: number;
}

export interface ParsedDxfSummary {
  units: string;
  unitToMeters: number;
  layers: Array<{ name: string; entityCount: number }>;
  entityCounts: Record<string, number>;
  blocks: Array<{ name: string; count: number }>;
  texts: string[];
  detectedRooms: ParsedDxfRoom[];
  detectedApartments: ParsedDxfApartment[];
  sumIndoorRoomsSqM: number | null;
  sumExteriorSpacesSqM: number | null;
  boundingBox: {
    minX: number;
    minY: number;
    maxX: number;
    maxY: number;
    width: number;
    height: number;
  } | null;
}

export interface TrainingStatus {
  baseCorpusQuestions: number;
  baseUniqueDrawings: number;
  customQuestionsCount: number;
  correctionsCount: number;
  recentCustomEntries: Array<{
    id: string;
    path: string;
    question: string;
    answer: string | number;
    unit: string | null;
    kind: 'uploaded' | 'correction';
    createdAt: string;
  }>;
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

export interface ProjectFileEntry {
  asset: LoadedAsset;
  parsedSummary: Record<string, unknown>;
  analysis: TakeoffAnalysis | null;
  chatMessages: ChatMessage[];
  matchedGroundTruth: GroundTruthQA[];
  calibrationNote: string | null;
  addedAt: string;
}
